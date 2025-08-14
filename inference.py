import os
import io
import csv
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

# == App Initialization ==
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")
LOG_FILE = "chat_logs.csv"

# == Model Wrapper ==
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get("clf")
        self.symptom_index = model_dict.get("symptom_index", {})
        self.disease_index = model_dict.get("disease_index", {})
        self.disease_info = model_dict.get("disease_info", {})
        self.vocab = list(symptom_embeddings.get("vocab", []))
        self.vectors = symptom_embeddings.get("vectors", None)
        if isinstance(self.vectors, np.ndarray):
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        canonical = [s for s in symptoms if s in self.symptom_index]
        x = np.zeros((1, len(self.symptom_index)), dtype=float)
        for s in canonical:
            idx = self.symptom_index.get(s)
            if idx is not None:
                x[0, idx] = 1.0

        try:
            if hasattr(self.clf, "predict_proba"):
                proba_vec = self.clf.predict_proba(x)
                top_idx = int(np.argmax(proba_vec[0]))
                proba = float(proba_vec[0, top_idx])
            else:
                top_idx = int(self.clf.predict(x)[0])
                proba = None
        except:
            top_idx, proba = 0, None

        condition = self.decode_condition(top_idx)
        info = self.disease_info.get(condition, {})
        return {
            "condition": condition,
            "proba": proba,
            "matched_symptoms": canonical,
            "advice": info.get("advice", "Stay hydrated and rest."),
            "precautions": info.get("precautions", [])
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"

def load_bundle() -> ModelBundle:
    model_dict = joblib.load(ARTIFACT_MODEL)
    npz = np.load(ARTIFACT_EMBED, allow_pickle=True)
    symptom_embeddings = {k: npz[k] for k in npz.files}
    return ModelBundle(model_dict, symptom_embeddings)

BUNDLE = load_bundle()

# == CSV Logging ==
def log_to_csv(patient, symptoms_entered, matched, result):
    header = ["timestamp", "name", "age", "sex", "symptoms_entered",
              "matched_symptoms", "predicted_condition", "probability", "advice"]
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            datetime.utcnow().isoformat(),
            patient.get("name", ""),
            patient.get("age", ""),
            patient.get("sex", ""),
            "; ".join(symptoms_entered),
            "; ".join(matched),
            result.get("condition", ""),
            f"{result.get('proba', ''):.2f}" if result.get("proba") is not None else "",
            result.get("advice", "")
        ])

# == State Management ==
def init_state():
    session.setdefault("stage", "ask_symptoms")
    session.setdefault("symptoms", [])
    session.setdefault("suggestion", None)
    session.setdefault("patient", {})
    session.setdefault("last_result", None)

# == Frontend UI (embedded) ==
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CareGuide AI</title>
  <style>
    body { font-family: system-ui; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width:900px; margin: auto; padding: 24px; }
    .card { background:#111827; border-radius:16px; padding:20px; }
    .chat { height:50vh; overflow:auto; background:#0b1220; padding:12px; border-radius:12px; }
    .msg { margin:8px 0; white-space: pre-wrap; }
    .ai { color:#a7f3d0; }
    .user { color:#93c5fd; text-align:right; }
    .row { display:flex; gap:8px; margin-top:12px; }
    input { flex:1; background:#0b1220; border:1px solid #1f2937; color:#e5e7eb; padding:10px; border-radius:10px; }
    button { background:#22c55e; color:#052e16; padding:10px 14px; border:none; border-radius:10px; font-weight:bold; cursor:pointer; }
    .download { background:#38bdf8; color:#06263a; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>CareGuide AI</h1>
      <div class="chat" id="chat"></div>
      <div class="row">
        <input id="input" placeholder="Type your reply here..." autocomplete="off"/>
        <button onclick="send()">Send</button>
      </div>
      <div class="row">
        <button class="download" onclick="downloadPdf()">Download prescription (PDF)</button>
      </div>
    </div>
  </div>
<script>
const chat = document.getElementById('chat'), input = document.getElementById('input');
function append(role, txt){const d = document.createElement('div'); d.className = 'msg '+role; d.textContent = txt; chat.appendChild(d); chat.scrollTop = chat.scrollHeight;}
async function boot(){const r=await fetch('/api/boot'); const j=await r.json(); append('ai', j.message);}
async function send(){
  const msg = input.value.trim(); if(!msg) return;
  append('user', msg); input.value = '';
  const r = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body: JSON.stringify({message: msg})});
  const j = await r.json(); append('ai', j.message);
}
async function downloadPdf(){window.location.href = '/download_pdf';}
boot();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    init_state()
    return render_template_string(HTML)

# == Routes ==
@app.route("/api/boot")
def api_boot():
    init_state()
    return jsonify({"message": "Hello! I'm CareGuide AI. Describe your symptoms separated by commas (e.g., 'fever, cough'). Type 'done' when finished."})

@app.route("/api/message", methods=["POST"])
def api_message():
    init_state()
    data = request.json or {}
    text = data.get("message", "").strip()
    stage = session["stage"]

    # Handle confirmation
    if session.get("suggestion"):
        if text.lower() in ("yes", "y"):
            suggestion = session.pop("suggestion")
            session["symptoms"].append(suggestion)
            session["stage"] = "ask_name"
            return jsonify({"message": f"Added '{suggestion}'. What is your full name?"})
        elif text.lower() in ("no", "n"):
            session.pop("suggestion")
            return jsonify({"message": "Okay, please re-enter your symptoms."})
        # else wait for valid confirmation

    if stage == "ask_symptoms":
        if text.lower() in ("done", "finish"):
            if not session["symptoms"]:
                return jsonify({"message": "Please add at least one symptom before typing 'done'."})
            session["stage"] = "ask_name"
            return jsonify({"message": "Got it. What is your full name?"})
        tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
        for term in tokens:
            if term not in session["symptoms"]:
                suggestion = suggest_nearest(term)
                if suggestion and suggestion != term:
                    session["suggestion"] = suggestion
                    return jsonify({"message": f"I didn’t recognize '{term}'. Did you mean '{suggestion}'? (yes/no)"})
                session["symptoms"].append(term)
        return jsonify({"message": "Noted: " + ", ".join(session["symptoms"]) + ". When finished, type 'done'."})

    if stage == "ask_name":
        session["patient"]["name"] = text
        session["stage"] = "ask_age"
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == "ask_age":
        session["patient"]["age"] = ''.join([c for c in text if c.isdigit()]) or text
        session["stage"] = "ask_sex"
        return jsonify({"message": "What is your sex? (male/female/other)"})

    if stage == "ask_sex":
        session["patient"]["sex"] = text.lower()
        symptoms_entered = session["symptoms"][:]
        result = BUNDLE.predict(symptoms_entered)
        log_to_csv(session["patient"], symptoms_entered, result["matched_symptoms"], result)
        session["last_result"] = result
        session["stage"] = "done"
        proba_txt = f" (confidence {result['proba']:.0%})" if result["proba"] is not None else ""
        extras = ""
        if result["precautions"]:
            extras = "\nPrecautions: " + ", ".join(result["precautions"])
        return jsonify({"message": f"Based on your symptoms, you may have: {result['condition']}{proba_txt}.\nAdvice: {result['advice']}{extras}"})

    return jsonify({"message": "Session complete. Refresh to start again."})

@app.route("/download_pdf")
def download_pdf():
    result = session.get("last_result")
    if not result:
        return "No diagnosis yet.", 400
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20*mm

    def line(text, step=7):
        nonlocal y
        c.drawString(20*mm, y, text)
        y -= step*mm

    c.setFont("Helvetica-Bold", 14)
    line("CareGuide AI — Prescription")
    c.setFont("Helvetica", 10)
    p = session.get("patient", {})
    line(f"Name: {p.get('name','N/A')}, Age: {p.get('age','N/A')}, Sex: {p.get('sex','N/A')}")
    line("Symptoms: " + ", ".join(session.get("symptoms", [])))
    line("")
    line(f"Condition: {result['condition']}")
    if result["proba"] is not None:
        line(f"Confidence: {result['proba']:.0%}")
    line("Advice:")
    for seg in result["advice"].split("\n"):
        line(seg)
    if result["precautions"]:
        line("Precautions:")
        for pz in result["precautions"]:
            line(f"• {pz}")
    c.showPage()
    c.save()
    buf.seek(0)
    fname = f"careguide_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/pdf")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def suggest_nearest(term: str) -> str:
    if not BUNDLE.vectors.any():
        return term
    v = embed_text(term)
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = BUNDLE.vectors @ v
    idx = int(np.argmax(sims))
    return BUNDLE.vocab[idx] if sims[idx] >= 0.45 else term

def embed_text(text: str, dim: int = 300) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
