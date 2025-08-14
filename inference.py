import os
import io
import json
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify, render_template_string, session, send_file
from flask_cors import CORS
from flask_session import Session
import joblib

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# Session config (persistent)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session_data'
app.config['SESSION_PERMANENT'] = False
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-please")
Session(app)

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")

# -----------------------------
# Model bundle
# -----------------------------
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get('clf')
        self.symptom_index = model_dict.get('symptom_index', {})
        self.disease_index = model_dict.get('disease_index', {})
        self.disease_info = model_dict.get('disease_info', {})
        self.vocab = list(symptom_embeddings.get('vocab', []))
        self.vectors = symptom_embeddings.get('vectors')
        if isinstance(self.vectors, np.ndarray):
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        canonical = [s for s in symptoms if s in self.symptom_index]
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
            "advice": info.get('advice', 'Stay hydrated and rest.'),
            "precautions": info.get('precautions', ['Rest well', 'Drink water']),
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        elif isinstance(self.disease_index, (list, tuple)):
            return self.disease_index[idx]
        return f"Class_{idx}"

def load_bundle():
    try:
        model_dict = joblib.load(ARTIFACT_MODEL)
    except Exception:
        model_dict = {}
    try:
        npz = np.load(ARTIFACT_EMBED, allow_pickle=True)
        symptom_embeddings = {k: npz[k] for k in npz.files}
    except Exception:
        symptom_embeddings = {}
    return ModelBundle(model_dict, symptom_embeddings)

BUNDLE = load_bundle()

# -----------------------------
# State helpers
# -----------------------------
def init_state():
    if 'stage' not in session:
        session['stage'] = 'ask_symptoms'
        session['symptoms'] = []
        session['patient'] = {}
        session['last_result'] = None

def reset_state():
    session.clear()
    init_state()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    init_state()
    return render_template_string("""
        <h1>CareGuide AI</h1>
        <p>Welcome! Use the API endpoints to interact with the assistant.</p>
    """)

@app.route("/api/boot")
def api_boot():
    reset_state()
    return jsonify({
        "message": "Hello! I'm CareGuide AI. Please describe your symptoms separated by commas (e.g., 'fever, cough'). Type 'done' when finished."
    })

@app.route("/api/message", methods=["POST"])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get("message") or "").strip().lower()

    stage = session['stage']

    if stage == "ask_symptoms":
        if text in ["done", "finish", "end"]:
            if not session['symptoms']:
                return jsonify({"message": "Please add at least one symptom before finishing."})
            session['stage'] = 'ask_name'
            return jsonify({"message": "Got it. What is your full name?"})
        else:
            tokens = [t.strip() for t in text.split(",") if t.strip()]
            session['symptoms'].extend(tokens)
            return jsonify({"message": "Noted: " + ", ".join(tokens) + ". Add more or type 'done' to continue."})

    elif stage == "ask_name":
        session['patient']['name'] = text
        session['stage'] = 'ask_age'
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    elif stage == "ask_age":
        digits = ''.join(ch for ch in text if ch.isdigit())
        session['patient']['age'] = int(digits) if digits else None
        session['stage'] = 'ask_sex'
        return jsonify({"message": "What is your sex? (male/female/other)"})

    elif stage == "ask_sex":
        session['patient']['sex'] = text
        result = BUNDLE.predict(session['symptoms'])
        session['last_result'] = {
            "patient": session['patient'],
            "symptoms": session['symptoms'],
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        session['stage'] = 'done'
        proba_txt = f" (confidence {result['proba']:.0%})" if result['proba'] is not None else ""
        precautions = ", ".join(result['precautions'])
        return jsonify({"message": f"Based on your symptoms, you may have {result['condition']}{proba_txt}.\nAdvice: {result['advice']}\nPrecautions: {precautions}"})

    return jsonify({"message": "Session complete. Restart if needed."})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
