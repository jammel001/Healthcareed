import os, io
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify, render_template_string, session, send_file
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-please")

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")

# -----------------------------
# Model Bundle
# -----------------------------
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get("clf")
        self.symptom_index = model_dict.get("symptom_index", {})
        self.disease_index = model_dict.get("disease_index", {})
        self.disease_info = model_dict.get("disease_info", {})
        self.vocab = list(symptom_embeddings.get("vocab", [])) if symptom_embeddings else []
        self.vectors = symptom_embeddings.get("vectors", None) if symptom_embeddings else None

        if isinstance(self.vectors, np.ndarray) and self.vectors.size > 0:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        canonical = self.match_symptoms(symptoms)
        x = np.zeros((1, len(self.symptom_index)), dtype=float)
        for s in canonical:
            idx = self.symptom_index.get(s)
            if idx is not None:
                x[0, idx] = 1.0

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
            "advice": info.get("advice", "Stay hydrated and seek medical attention if symptoms worsen."),
            "precautions": info.get("precautions", ["Rest well", "Stay hydrated"]),
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"

    def match_symptoms(self, raw_symptoms: List[str]) -> List[str]:
        if self.vectors is None or not self.vocab:
            return [t for t in raw_symptoms if t in self.symptom_index]

        matched = []
        for s in raw_symptoms:
            v = embed_text(s)
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = self.vectors @ v
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= 0.45:
                matched.append(self.vocab[best_idx])
        return list(set(matched))


def embed_text(text: str, dim: int = 300) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)


def load_bundle() -> ModelBundle:
    try:
        model_dict = joblib.load(ARTIFACT_MODEL)
    except:
        model_dict = {}
    try:
        npz = np.load(ARTIFACT_EMBED, allow_pickle=True)
        symptom_embeddings = {k: npz[k] for k in npz.files}
    except:
        symptom_embeddings = {}
    return ModelBundle(model_dict, symptom_embeddings)


BUNDLE = load_bundle()

# -----------------------------
# State Helpers
# -----------------------------
def init_state():
    session.setdefault("stage", "greet")
    session.setdefault("patient", {})
    session.setdefault("symptoms", [])
    session.setdefault("last_result", None)
    session.setdefault("pending_suggestion", None)
    session.modified = True

# -----------------------------
# HTML Template
# -----------------------------
HOME_HTML = """
<!doctype html>
<html>
<head>
  <title>CareGuide AI</title>
  <style>
    body { font-family: system-ui; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 900px; margin: auto; padding: 24px; }
    .card { background: #111827; border-radius: 16px; padding: 20px; }
    .chat { height: 50vh; overflow: auto; background: #0b1220; padding: 12px; border-radius: 12px; }
    .msg { margin: 8px 0; }
    .ai { color: #a7f3d0; }
    .user { color: #93c5fd; text-align: right; }
    input { flex: 1; padding: 10px; border-radius: 10px; border: none; }
    button { background: #22c55e; padding: 10px 14px; border-radius: 10px; font-weight: bold; cursor: pointer; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>CareGuide AI</h1>
    <div class="chat" id="chat"></div>
    <div style="display:flex; gap:8px; margin-top:12px;">
      <input id="input" placeholder="Type your reply here..." />
      <button onclick="send()">Send</button>
    </div>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
function append(role, text){
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
async function boot(){
  const r = await fetch('/api/boot');
  const j = await r.json();
  append('ai', j.message);
}
async function send(){
  const msg = input.value.trim();
  if(!msg) return;
  append('user', msg);
  input.value = '';
  const r = await fetch('/api/message', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msg })
  });
  const j = await r.json();
  append('ai', j.message);
}
boot();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    init_state()
    return render_template_string(HOME_HTML)

# -----------------------------
# Conversation API
# -----------------------------
@app.route("/api/boot")
def api_boot():
    init_state()
    session["stage"] = "ask_symptoms"
    return jsonify({
        "message": (
            "Hello! I'm CareGuide AI.\n"
            "Please describe your symptoms separated by commas (e.g., 'fever, cough').\n"
            "Type 'done' when finished."
        )
    })


@app.route("/api/message", methods=["POST"])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get("message") or "").strip()
    stage = session.get("stage", "ask_symptoms")

    pending = session.get("pending_suggestion")
    if pending:
        if text.lower() in ("yes", "y"):
            syms = session.get("symptoms", [])
            if pending not in syms:
                syms.append(pending)
            session["symptoms"] = syms
            session["pending_suggestion"] = None
            session.modified = True
            return jsonify({"message": "Added. Any other symptoms? Type 'done' when finished."})
        elif text.lower() in ("no", "n"):
            session["pending_suggestion"] = None
            session.modified = True
            return jsonify({"message": "Okay, please add a different symptom."})

    if stage == "ask_symptoms":
        if text.lower() in ("done", "finish", "end", "no"):
            if not session.get("symptoms"):
                return jsonify({"message": "I don’t have any symptoms yet. Please add at least one before typing 'done'."})
            session["stage"] = "ask_name"
            session.modified = True
            return jsonify({"message": "Got it. What is your full name?"})
        else:
            tokens = [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]
            for term in tokens:
                if term not in BUNDLE.symptom_index:
                    suggestion = suggest_nearest(term, BUNDLE)
                    if suggestion:
                        session["pending_suggestion"] = suggestion
                        session.modified = True
                        return jsonify({"message": f"I didn't recognize '{term}'. Did you mean '{suggestion}'? (yes/no)"})
            syms = session.get("symptoms", [])
            for s in tokens:
                if s not in syms:
                    syms.append(s)
            session["symptoms"] = syms
            session.modified = True
            return jsonify({"message": "Noted: " + ", ".join(tokens) + ". Add more or type 'done' to continue."})

    if stage == "ask_name":
        session["patient"]["name"] = text
        session["stage"] = "ask_age"
        session.modified = True
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == "ask_age":
        session["patient"]["age"] = text
        session["stage"] = "ask_sex"
        session.modified = True
        return jsonify({"message": "What is your sex? (male/female/other)"})

    if stage == "ask_sex":
        session["patient"]["sex"] = text.lower()
        result = BUNDLE.predict(session.get("symptoms", []))
        session["last_result"] = {
            "patient": session.get("patient", {}),
            "symptoms": session.get("symptoms", []),
            "result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        session["stage"] = "done"
        session.modified = True
        proba_txt = f" (confidence {result['proba']:.0%})" if result["proba"] is not None else ""
        precautions = result.get("precautions", [])
        extra = ("\nPrecautions: " + ", ".join(precautions)) if precautions else ""
        return jsonify({"message": f"Based on your symptoms, you may have: {result['condition']}{proba_txt}.\nAdvice: {result['advice']}{extra}"})

    return jsonify({"message": "We’ve completed your assessment. Refresh to start again."})


def suggest_nearest(term: str, bundle: ModelBundle) -> str:
    if not bundle.vocab:
        return ""
    v = embed_text(term)
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = bundle.vectors @ v
    idx = int(np.argmax(sims))
    return bundle.vocab[idx] if sims[idx] >= 0.45 else ""


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
