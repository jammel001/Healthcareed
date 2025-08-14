import os, io, json
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-please")

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")

# -----------------------------
# Model bundle
# -----------------------------
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get("clf", None)
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
        if self.clf is None or not self.symptom_index:
            return {
                "condition": "Insufficient model artifacts",
                "proba": 0.0,
                "matched_symptoms": [],
                "advice": "Model not available. Please retrain.",
                "precautions": []
            }
        canonical = self.match_symptoms(symptoms)
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
        except Exception:
            top_idx = 0
            proba = None
        condition = self.decode_condition(top_idx)
        info = self.disease_info.get(condition, {})
        return {
            "condition": condition,
            "proba": proba,
            "matched_symptoms": canonical,
            "advice": info.get("advice", "Stay hydrated and seek medical attention if needed."),
            "precautions": info.get("precautions", ["Stay hydrated", "Rest", "Monitor your symptoms"])
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"

    def match_symptoms(self, raw_symptoms: List[str]) -> List[str]:
        if not raw_symptoms:
            return []
        tokens = [t.strip().lower() for s in raw_symptoms for t in s.replace(';', ',').split(',') if t.strip()]
        return [t for t in tokens if t in self.symptom_index]

def load_bundle() -> ModelBundle:
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
    session.setdefault("stage", "ask_symptoms")
    session.setdefault("patient", {})
    session.setdefault("symptoms", [])
    session.setdefault("last_result", None)

def reset_state():
    session.clear()
    init_state()

# -----------------------------
# UI
# -----------------------------
HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CareGuide AI</title>
  <style>
    body{font-family:system-ui;background:#0f172a;color:#e2e8f0;margin:0}
    .wrap{max-width:900px;margin:auto;padding:24px}
    .card{background:#111827;border-radius:16px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    .chat{height:50vh;overflow:auto;border:1px solid #1f2937;border-radius:12px;padding:12px;background:#0b1220}
    .msg{margin:8px 0;}
    .ai{color:#a7f3d0}
    .user{color:#93c5fd;text-align:right}
    input{flex:1;background:#0b1220;border:1px solid #1f2937;color:#e5e7eb;padding:10px;border-radius:10px}
    button{background:#22c55e;color:#052e16;border:none;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer}
    .download{background:#38bdf8;color:#06263a}
    .row{display:flex;gap:8px;margin-top:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>CareGuide AI</h1>
      <div class="chat" id="chat"></div>
      <div class="row">
        <input id="input" placeholder="Type here..." />
        <button onclick="send()">Send</button>
      </div>
      <div class="row">
        <button class="download" onclick="downloadPdf()">Download PDF</button>
      </div>
    </div>
  </div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
function append(role, text){const div=document.createElement('div');div.className='msg '+role;div.textContent=text;chat.appendChild(div);chat.scrollTop=chat.scrollHeight;}
async function boot(){const r=await fetch('/api/boot');const j=await r.json();append('ai', j.message);}
async function send(){const msg=input.value.trim();if(!msg) return;append('user', msg);input.value='';const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});const j=await r.json();append('ai', j.message);}
async function downloadPdf(){window.location.href='/download_pdf';}
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
# API
# -----------------------------
@app.route("/api/boot")
def api_boot():
    reset_state()
    return jsonify({"message": "Hello! I'm CareGuide AI. Please describe your symptoms separated by commas (e.g., 'fever, cough'). Type 'done' when finished."})

@app.route("/api/message", methods=["POST"])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get("message") or "").strip()
    stage = session["stage"]

    if stage == "ask_symptoms":
        if text.lower() in ("done", "finish"):
            if not session["symptoms"]:
                return jsonify({"message": "Please add at least one symptom before finishing."})
            session["stage"] = "ask_name"
            return jsonify({"message": "Got it. What is your full name?"})
        else:
            tokens = [t.strip().lower() for t in text.replace(";", ",").split(",") if t.strip()]
            for t in tokens:
                if t not in session["symptoms"]:
                    session["symptoms"].append(t)
            return jsonify({"message": "Noted: " + ", ".join(tokens) + ". Add more or type 'done' to continue."})

    if stage == "ask_name":
        session["patient"]["name"] = text
        session["stage"] = "ask_age"
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == "ask_age":
        digits = "".join([ch for ch in text if ch.isdigit()])
        session["patient"]["age"] = int(digits) if digits else None
        session["stage"] = "ask_sex"
        return jsonify({"message": "What is your sex? (male/female/other)"})

    if stage == "ask_sex":
        session["patient"]["sex"] = text.lower()
        result = BUNDLE.predict(session["symptoms"])
        session["last_result"] = {
            "patient": session["patient"],
            "symptoms": session["symptoms"],
            "result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        session["stage"] = "done"
        proba_txt = f" (confidence {result['proba']:.0%})" if result["proba"] is not None else ""
        precautions = result.get("precautions", [])
        extra = ("\nPrecautions: " + ", ".join(precautions)) if precautions else ""
        return jsonify({"message": f"Thank you {session['patient'].get('name','')}. Based on your symptoms, you may have: {result['condition']}{proba_txt}.\n\nAdvice: {result['advice']}{extra}\n\nYou can download your PDF now or refresh to start a new session."})

    return jsonify({"message": "Session finished. Please refresh to start over."})

@app.route("/download_pdf")
def download_pdf():
    lr = session.get("last_result")
    if not lr:
        return jsonify({"error": "No result yet"}), 400
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20*mm
    def line(txt, step=7):
        nonlocal y
        c.drawString(20*mm, y, txt)
        y -= step*mm
    c.setFont("Helvetica-Bold", 14)
    line("CareGuide AI — Well-Being Guidance")
    c.setFont("Helvetica", 10)
    p = lr["patient"]
    line(f"Patient: {p.get('name','N/A')}    Age: {p.get('age','N/A')}    Sex: {p.get('sex','N/A')}")
    line(f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    line("")
    line("Reported Symptoms:")
    for s in lr["symptoms"]:
        line(f" • {s}")
    r = lr["result"]
    line("")
    line(f"Likely Condition: {r['condition']}")
    if r["proba"] is not None:
        line(f"Confidence: {r['proba']:.0%}")
    line("")
    line("Advice:")
    from textwrap import wrap
    for chunk in wrap(r["advice"], 90):
        line(chunk)
    if r.get("precautions"):
        line("")
        line("Precautions:")
        for pz in r["precautions"]:
            line(f" • {pz}")
    c.showPage()
    c.save()
    buf.seek(0)
    filename = f"careguide_prescription_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
