import os, io
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib

# Optional fuzzy matching for typos
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

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
        self.clf = model_dict.get('clf', None)
        self.symptom_index = model_dict.get('symptom_index', {})
        self.disease_index = model_dict.get('disease_index', {})
        self.disease_info = model_dict.get('disease_info', {})

        # Embeddings for symptom matching
        self.vocab = list(symptom_embeddings.get('vocab', [])) if symptom_embeddings else []
        self.vectors = symptom_embeddings.get('vectors', None) if symptom_embeddings else None
        if isinstance(self.vectors, np.ndarray) and self.vectors.size > 0:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        if self.clf is None or not self.symptom_index:
            return {
                "condition": "Model not available",
                "proba": None,
                "matched_symptoms": [],
                "advice": "Please redeploy with a trained model.",
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
            top_idx, proba = 0, None

        condition = self.decode_condition(top_idx)
        info = self.disease_info.get(condition, {})
        advice = info.get('advice', 'Stay hydrated and rest.')
        precautions = info.get('precautions', ['Stay hydrated', 'Rest well'])

        return {
            "condition": condition,
            "proba": proba,
            "matched_symptoms": canonical,
            "advice": advice,
            "precautions": precautions
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"

    def match_symptoms(self, raw_symptoms: List[str]) -> List[str]:
        if self.vectors is None or not self.vocab:
            return [t for s in raw_symptoms for t in s.split(',') if t in self.symptom_index]

        matched = []
        for s in raw_symptoms:
            for term in [w.strip().lower() for w in s.split(',') if w.strip()]:
                v = embed_text(term)
                if v is None: 
                    continue
                v = v / (np.linalg.norm(v) + 1e-12)
                sims = self.vectors @ v
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= 0.45:
                    matched.append(self.vocab[best_idx])

        return list(dict.fromkeys(matched))  # dedupe

def embed_text(text: str, dim: int = 300) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)

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
# Conversation state
# -----------------------------
def init_state():
    session.setdefault('stage', 'greet')
    session.setdefault('patient', {})
    session.setdefault('symptoms', [])
    session.setdefault('last_result', None)
    session.setdefault('pending_suggestion', None)

# -----------------------------
# Suggestion helper
# -----------------------------
def suggest_nearest(term: str, bundle: ModelBundle) -> str:
    if fuzz:
        candidates = list(bundle.symptom_index.keys())
        best, score = max(((c, fuzz.ratio(term, c))) for c in candidates)
        return best if score >= 70 else ""
    return ""

# -----------------------------
# Frontend
# -----------------------------
HOME_HTML = """
<!doctype html>
<html>
<head>
  <title>CareGuide AI</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body{font-family:system-ui;background:#0f172a;color:#e2e8f0;margin:0}
    .wrap{max-width:900px;margin:0 auto;padding:24px}
    .card{background:#111827;padding:20px;border-radius:16px}
    .chat{height:50vh;overflow:auto;background:#0b1220;padding:12px;border-radius:12px}
    .msg{margin:8px 0}
    .ai{color:#a7f3d0}
    .user{color:#93c5fd;text-align:right}
    .row{display:flex;gap:8px;margin-top:12px}
    input{flex:1;background:#0b1220;color:#fff;padding:10px;border:1px solid #1f2937;border-radius:10px}
    button{background:#22c55e;color:#052e16;border:none;padding:10px 14px;border-radius:10px;font-weight:700}
    .download{background:#38bdf8;color:#06263a;display:none}
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
      <button class="download" onclick="downloadPdf()">Download last prescription (PDF)</button>
    </div>
  </div>
</div>
<script>
const chat=document.getElementById('chat');const input=document.getElementById('input');
function append(role,text){const div=document.createElement('div');div.className='msg '+role;div.textContent=text;chat.appendChild(div);chat.scrollTop=chat.scrollHeight;}
async function boot(){const r=await fetch('/api/boot');const j=await r.json();append('ai',j.message);}
async function send(){
  const msg=input.value.trim();if(!msg)return;
  append('user',msg);input.value='';
  append('ai',"Thinking...");
  const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
  const j=await r.json();chat.lastChild.remove();
  append('ai',j.message);
  if(j.pdf_ready)document.querySelector('.download').style.display='block';
}
async function downloadPdf(){window.location.href='/download_pdf';}
boot();
</script>
</body>
</html>
"""

@app.route('/')
def home():
    init_state()
    return render_template_string(HOME_HTML)

# -----------------------------
# API
# -----------------------------
@app.route('/api/boot')
def api_boot():
    init_state()
    session['stage'] = 'ask_symptoms'
    return jsonify({"message": "Hello! Please list your symptoms separated by commas."})

@app.route('/api/message', methods=['POST'])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get('message') or '').strip()
    stage = session['stage']

    # Handle pending suggestion
    if session.get('pending_suggestion'):
        if text.lower() in ['yes','y']:
            session['symptoms'].append(session['pending_suggestion'])
            session['pending_suggestion'] = None
            return jsonify({"message": "Added. Any other symptoms? Or type 'done'."})
        elif text.lower() in ['no','n']:
            session['pending_suggestion'] = None
            return jsonify({"message": "Okay, please type the correct symptom."})

    if stage == 'ask_symptoms':
        if text.lower() == 'done':
            if not session['symptoms']:
                return jsonify({"message": "Please add at least one symptom before continuing."})
            session['stage'] = 'ask_name'
            return jsonify({"message": "Got it. What’s your full name?"})
        tokens = [t.strip().lower() for t in text.split(',')]
        for term in tokens:
            if term in BUNDLE.symptom_index:
                session['symptoms'].append(term)
            else:
                suggestion = suggest_nearest(term, BUNDLE)
                if suggestion:
                    session['pending_suggestion'] = suggestion
                    return jsonify({"message": f"Did you mean '{suggestion}'?"})
        return jsonify({"message": "Symptoms noted. Add more or type 'done'."})

    if stage == 'ask_name':
        session['patient']['name'] = text
        session['stage'] = 'ask_age'
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == 'ask_age':
        session['patient']['age'] = int(''.join(filter(str.isdigit, text)))
        session['stage'] = 'ask_gender'
        return jsonify({"message": "What is your sex? (male/female/other)"})

    if stage == 'ask_gender':
        session['patient']['gender'] = text.lower()
        result = BUNDLE.predict(session['symptoms'])
        session['last_result'] = {
            "patient": session['patient'],
            "symptoms": session['symptoms'],
            "result": result
        }
        cond = result['condition']
        proba_txt = f" (confidence {result['proba']:.0%})" if result['proba'] else ""
        msg = f"Based on your symptoms, you may have: {cond}{proba_txt}.\n\nAdvice: {result['advice']}\nPrecautions: {', '.join(result['precautions'])}"
        return jsonify({"message": msg, "pdf_ready": True})

    return jsonify({"message": "Session complete. Refresh to start again."})

@app.route('/download_pdf')
def download_pdf():
    lr = session.get('last_result')
    if not lr:
        return jsonify({"error": "No result available."}), 400
    buf = io.BytesIO()
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20*mm
    def line(txt, step=7): nonlocal y; c.drawString(20*mm, y, txt); y -= step*mm
    c.setFont("Helvetica-Bold", 14); line("CareGuide AI — Prescription")
    c.setFont("Helvetica", 10)
    p = lr['patient']
    line(f"Patient: {p.get('name','N/A')}    Age: {p.get('age','N/A')}    Sex: {p.get('gender','N/A')}")
    line(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"); line("")
    line("Symptoms:"); [line(f" - {s}") for s in lr['symptoms']]
    r = lr['result']
    line(f"Likely Condition: {r['condition']}")
    if r['proba']: line(f"Confidence: {r['proba']:.0%}")
    line("Advice:"); [line(f" - {t}") for t in r['advice'].split('. ') if t]
    line("Precautions:"); [line(f" - {p}") for p in r['precautions']]
    c.showPage(); c.save(); buf.seek(0)
    filename = f"careguide_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route('/health')
def health(): 
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
