import os, io, csv
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string, session, send_file
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ==============================
# App setup
# ==============================
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")
LOG_PATH = "logs/conversations.csv"
os.makedirs("logs", exist_ok=True)

# ==============================
# Model bundle
# ==============================
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get('clf')
        self.symptom_index = model_dict.get('symptom_index', {})
        self.disease_index = model_dict.get('disease_index', {})
        self.disease_info = model_dict.get('disease_info', {})

        self.vocab = list(symptom_embeddings.get('vocab', [])) if symptom_embeddings else []
        self.vectors = symptom_embeddings.get('vectors') if symptom_embeddings else None
        if isinstance(self.vectors, np.ndarray) and self.vectors.size > 0:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        canonical = self.match_symptoms(symptoms)
        x = np.zeros((1, len(self.symptom_index)))
        for s in canonical:
            idx = self.symptom_index.get(s)
            if idx is not None:
                x[0, idx] = 1

        if hasattr(self.clf, "predict_proba"):
            proba_vec = self.clf.predict_proba(x)
            top_idx = int(np.argmax(proba_vec[0]))
            proba = float(proba_vec[0, top_idx])
        else:
            top_idx = int(self.clf.predict(x)[0])
            proba = None

        condition = self.decode_condition(top_idx)
        info = self.disease_info.get(condition, {})
        return {
            "condition": condition,
            "proba": proba,
            "advice": info.get('advice', 'Stay hydrated, rest well.'),
            "precautions": info.get('precautions', [])
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        return f"Class_{idx}"

    def match_symptoms(self, raw: List[str]) -> List[str]:
        if self.vectors is None or not self.vocab:
            return [t for t in raw if t in self.symptom_index]
        matched = []
        for term in raw:
            v = embed_text(term)
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = self.vectors @ v
            idx = int(np.argmax(sims))
            if sims[idx] >= 0.45:
                matched.append(self.vocab[idx])
        return list(set(matched))

def embed_text(text: str, dim: int = 300) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)

def load_bundle() -> ModelBundle:
    model_dict = joblib.load(ARTIFACT_MODEL)
    npz = np.load(ARTIFACT_EMBED, allow_pickle=True)
    symptom_embeddings = {k: npz[k] for k in npz.files}
    return ModelBundle(model_dict, symptom_embeddings)

BUNDLE = load_bundle()

# ==============================
# Conversation helpers
# ==============================
def init_state():
    session.setdefault("stage", "ask_symptoms")
    session.setdefault("symptoms", [])
    session.setdefault("pending_suggestion", None)
    session.setdefault("patient", {})
    session.setdefault("last_result", None)

def log_interaction(user_msg, bot_msg):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "stage", "user_message", "bot_message", "symptoms", "prediction", "proba"])
        pred = session.get("last_result", {}).get("result", {})
        writer.writerow([
            datetime.utcnow().isoformat(),
            session.get("stage"),
            user_msg,
            bot_msg,
            "; ".join(session.get("symptoms", [])),
            pred.get("condition"),
            pred.get("proba")
        ])

def suggest_nearest(term: str) -> str:
    if not BUNDLE.vectors.any():
        return ""
    v = embed_text(term)
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = BUNDLE.vectors @ v
    idx = int(np.argmax(sims))
    return BUNDLE.vocab[idx] if sims[idx] >= 0.45 else ""

# ==============================
# UI
# ==============================
HOME_HTML = """<!doctype html>..."""  # keep your existing style here

@app.route("/")
def home():
    init_state()
    return render_template_string(HOME_HTML)

# ==============================
# API
# ==============================
@app.route("/api/boot")
def api_boot():
    init_state()
    msg = "Hello! I'm CareGuide AI. Please describe your symptoms separated by commas (e.g., 'fever, cough'). Type 'done' when finished."
    log_interaction("", msg)
    return jsonify({"message": msg})

@app.route("/api/message", methods=["POST"])
def api_message():
    init_state()
    text = request.json.get("message", "").strip().lower()
    stage = session["stage"]

    # Handle pending yes/no
    if session.get("pending_suggestion"):
        if text in ("yes", "y"):
            syms = session["symptoms"]
            if session["pending_suggestion"] not in syms:
                syms.append(session["pending_suggestion"])
            session["pending_suggestion"] = None
            bot_msg = "Added. Any more symptoms? Type 'done' when finished."
            log_interaction(text, bot_msg)
            return jsonify({"message": bot_msg})
        elif text in ("no", "n"):
            session["pending_suggestion"] = None
            bot_msg = "Okay, please re-enter your symptom."
            log_interaction(text, bot_msg)
            return jsonify({"message": bot_msg})

    if stage == "ask_symptoms":
        if text in ("done", "finish", "end"):
            if not session["symptoms"]:
                bot_msg = "Please add at least one symptom before finishing."
                log_interaction(text, bot_msg)
                return jsonify({"message": bot_msg})
            session["stage"] = "ask_name"
            bot_msg = "Got it. What is your full name?"
            log_interaction(text, bot_msg)
            return jsonify({"message": bot_msg})

        tokens = [t.strip() for t in text.split(",") if t.strip()]
        for term in tokens:
            if term in BUNDLE.symptom_index:
                if term not in session["symptoms"]:
                    session["symptoms"].append(term)
            else:
                suggestion = suggest_nearest(term)
                if suggestion:
                    session["pending_suggestion"] = suggestion
                    bot_msg = f"I didn't recognize '{term}'. Did you mean '{suggestion}'? (yes/no)"
                    log_interaction(text, bot_msg)
                    return jsonify({"message": bot_msg})

        bot_msg = "Noted: " + ", ".join(session["symptoms"]) + ". Add more or type 'done' to continue."
        log_interaction(text, bot_msg)
        return jsonify({"message": bot_msg})

    if stage == "ask_name":
        session["patient"]["name"] = text
        session["stage"] = "ask_age"
        bot_msg = f"Thanks {text}. How old are you?"
        log_interaction(text, bot_msg)
        return jsonify({"message": bot_msg})

    if stage == "ask_age":
        session["patient"]["age"] = ''.join(ch for ch in text if ch.isdigit())
        session["stage"] = "ask_sex"
        bot_msg = "What is your sex? (male/female/other)"
        log_interaction(text, bot_msg)
        return jsonify({"message": bot_msg})

    if stage == "ask_sex":
        session["patient"]["sex"] = text
        result = BUNDLE.predict(session["symptoms"])
        session["last_result"] = {"result": result, "patient": session["patient"]}
        session["stage"] = "done"
        proba_txt = f" (confidence {result['proba']:.0%})" if result["proba"] else ""
        bot_msg = f"Based on your symptoms, you may have: {result['condition']}{proba_txt}. Advice: {result['advice']}"
        log_interaction(text, bot_msg)
        return jsonify({"message": bot_msg})

    bot_msg = "We’ve completed your assessment."
    log_interaction(text, bot_msg)
    return jsonify({"message": bot_msg})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
