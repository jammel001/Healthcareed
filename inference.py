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
        self.clf = model_dict.get('clf', None)
        self.symptom_index = model_dict.get('symptom_index', {})
        self.disease_index = model_dict.get('disease_index', {})
        self.disease_info = model_dict.get('disease_info', {})
        # embeddings
        self.vocab = list(symptom_embeddings.get('vocab', [])) if symptom_embeddings else []
        self.vectors = symptom_embeddings.get('vectors', None) if symptom_embeddings else None
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
                "advice": "Model or indices not found. Please rebuild and redeploy.",
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
            if hasattr(self.clf, "decision_function"):
                scores = self.clf.decision_function(x)
                top_idx = int(np.argmax(scores[0]))
                proba = None
            else:
                top_idx = 0
                proba = None

        condition = self.decode_condition(top_idx)
        info = self.disease_info.get(condition, {}) if isinstance(self.disease_info, dict) else {}
        advice = info.get('advice', 'Stay hydrated, rest, and monitor your symptoms. Seek medical help if they worsen.')
        precautions = info.get('precautions', ['Stay hydrated', 'Rest well', 'Monitor temperature'])

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
            tokens = [t.strip().lower() for s in raw_symptoms for t in s.replace(';', ',').split(',')]
            return [t for t in tokens if t in self.symptom_index]

        matched = []
        for s in raw_symptoms:
            terms = [w.strip().lower() for w in s.replace(';', ',').split(',') if w.strip()]
            for term in terms:
                v = embed_text(term)
                if v is None: 
                    continue
                v = v / (np.linalg.norm(v) + 1e-12)
                sims = self.vectors @ v
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= 0.45:
                    matched.append(self.vocab[best_idx])

        # dedupe
        seen, canonical = set(), []
        for m in matched:
            if m not in seen:
                seen.add(m)
                canonical.append(m)
        if not canonical:
            tokens = [t.strip().lower() for s in raw_symptoms for t in s.replace(';', ',').split(',')]
            canonical = [t for t in tokens if t in self.symptom_index]
        return canonical

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
# Conversation state helpers
# -----------------------------
def init_state():
    session.setdefault('stage', 'greet')
    session.setdefault('patient', {})
    session.setdefault('symptoms', [])
    session.setdefault('last_result', None)
    session.setdefault('pending_suggestion', None)

def reset_state():
    session.clear()
    init_state()

# -----------------------------
# Minimal UI (optional)
# -----------------------------
HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CareGuide AI</title>
  <style>
    body{font-family: system-ui; margin:0; background:#0f172a; color:#e2e8f0}
    .wrap{max-width:900px; margin:0 auto; padding:24px}
    .card{background:#111827; border-radius:16px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.25)}
    h1{margin:0 0 8px 0}
    .attribution{font-size:14px; opacity:.8; margin-bottom:12px}
    .chat{height:50vh; overflow:auto; border:1px solid #1f2937; border-radius:12px; padding:12px; background:#0b1220}
    .msg{margin:8px 0;}
    .ai{color:#a7f3d0}
    .user{color:#93c5fd; text-align:right}
    .row{display:flex; gap:8px; margin-top:12px}
    input{flex:1; background:#0b1220; border:1px solid #1f2937; color:#e5e7eb; padding:10px; border-radius:10px}
    button{background:#22c55e; color:#052e16; border:none; padding:10px 14px; border-radius:10px; font-weight:700; cursor:pointer}
    .download{background:#38bdf8; color:#06263a}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>CareGuide AI</h1>
      <div class="attribution">Built by <strong>Bara'u Magaji</strong>, <strong>Aliyu Muhammad Abdul</strong>, and <strong>Aliyu Biniyaminu</strong>.</div>
      <div class="chat" id="chat"></div>
      <div class="row">
        <input id="input" placeholder="Type your reply here..." />
        <button onclick="send()">Send</button>
      </div>
      <div class="row">
        <button class="download" onclick="downloadPdf()">Download last prescription (PDF)</button>
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
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ message: msg })
  });
  const j = await r.json();
  if(j.ask_confirm && j.suggestion){
    append('ai', j.message + `\nDid you mean: "${j.suggestion}"? (yes/no)`);
  } else {
    append('ai', j.message);
  }
}

async function downloadPdf(){
  const r = await fetch('/api/has_result');
  const j = await r.json();
  if(!j.ok){
    append('ai', 'No prescription available yet. Please complete a prediction first.');
    return;
  }
  window.location.href = '/download_pdf';
}

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
# Conversation API
# -----------------------------
@app.route('/api/boot')
def api_boot():
    init_state()
    # Order requested: greet + show builders → ask for feelings first → later ask details
    msg = (
        "Hello! I'm CareGuide AI.\n"
        "This assistant was built by Bara'u Magaji, Aliyu Muhammad Abdul, and Aliyu Biniyaminu.\n\n"
        "Please describe how you're feeling (e.g., 'fever, sore throat, headache'). "
        "You can list multiple symptoms separated by commas. When you're done, type 'done'."
    )
    session['stage'] = 'ask_symptoms'  # start with symptoms
    return jsonify({"message": msg})

@app.route('/api/message', methods=['POST'])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get('message') or '').strip()
    stage = session.get('stage', 'ask_symptoms')

    # Handle pending yes/no for suggestion
    pending = session.get('pending_suggestion')
    if pending:
        if text.lower() in ('yes', 'y'):
            syms = session.get('symptoms', [])
            if pending not in syms:
                syms.append(pending)
            session['symptoms'] = syms
            session['pending_suggestion'] = None
            return jsonify({"message": "Thanks, added. Any other symptoms? If you're done, type 'done'."})
        elif text.lower() in ('no', 'n'):
            session['pending_suggestion'] = None
            return jsonify({"message": "Okay, please rephrase or add a different symptom."})
        # otherwise continue below

    if stage == 'ask_symptoms':
        if text.lower() in ('done', 'finish', 'end', 'no'):
            if not session.get('symptoms'):
                return jsonify({"message": "I don’t have any symptoms yet. Please add at least one before typing 'done'."})
            session['stage'] = 'ask_name'
            return jsonify({"message": "Got it. Before I advise you, may I have your full name?"})
        else:
            # extract tokens and suggest nearest if unknown
            tokens = [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]
            accepted = []
            for term in tokens:
                if term in BUNDLE.symptom_index:
                    accepted.append(term)
                else:
                    suggestion = suggest_nearest(term, BUNDLE)
                    if suggestion:
                        session['pending_suggestion'] = suggestion
                        return jsonify({
                            "message": f"I didn't recognize '{term}'.",
                            "ask_confirm": True,
                            "suggestion": suggestion
                        })
            if accepted:
                syms = session.get('symptoms', [])
                for a in accepted:
                    if a not in syms:
                        syms.append(a)
                session['symptoms'] = syms
                return jsonify({"message": "Noted: " + ", ".join(accepted) + ". Add more or type 'done' to continue."})
            return jsonify({"message": "Thanks. Add more symptoms or type 'done' to proceed."})

    if stage == 'ask_name':
        session.setdefault('patient', {})['name'] = text
        session['stage'] = 'ask_age'
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == 'ask_age':
        age_num = ''.join(ch for ch in text if ch.isdigit())
        session['patient']['age'] = int(age_num) if age_num.isdigit() else None
        session['stage'] = 'ask_gender'
        return jsonify({"message": "What is your gender? (male/female/other)"})

    if stage == 'ask_gender':
        session['patient']['gender'] = text.lower()
        # Predict now
        syms = session.get('symptoms', [])
        result = BUNDLE.predict(syms)
        session['last_result'] = {
            "patient": session.get('patient', {}),
            "symptoms": syms,
            "result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        session['stage'] = 'done'
        cond = result['condition']
        proba_txt = f" (confidence {result['proba']:.0%})" if result['proba'] is not None else ""
        advice = result['advice']
        precautions = result.get('precautions', [])
        extra = ("\nPrecautions: " + ", ".join(precautions)) if precautions else ""
        return jsonify({"message":
            f"Thank you. Based on your symptoms, you may be experiencing: {cond}{proba_txt}.\n\n"
            f"Advice: {advice}{extra}\n\n"
            f"If you want, click 'Download last prescription (PDF)'."
        })

    return jsonify({"message": "We’ve completed your assessment. You can refresh to start over."})

def suggest_nearest(term: str, bundle: ModelBundle) -> str:
    if bundle.vectors is None or not bundle.vocab:
        # simple fallback: character Jaccard
        candidates = list(bundle.symptom_index.keys())
        term_l = term.lower()
        best, best_score = "", 0.0
        for c in candidates:
            score = jaccard_chars(term_l, c.lower())
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= 0.5 else ""
    v = embed_text(term)
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = bundle.vectors @ v
    idx = int(np.argmax(sims))
    return bundle.vocab[idx] if sims[idx] >= 0.45 else ""

def jaccard_chars(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

@app.route('/api/has_result')
def api_has_result():
    return jsonify({"ok": bool(session.get('last_result'))})

@app.route('/download_pdf')
def download_pdf():
    lr = session.get('last_result')
    if not lr:
        return jsonify({"error": "No result available."}), 400

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20*mm

    def line(txt, step=7):
        nonlocal y
        c.drawString(20*mm, y, txt)
        y -= step*mm

    c.setTitle("CareGuide AI — Prescription")
    c.setFont("Helvetica-Bold", 14); line("CareGuide AI — Well-Being Guidance")
    c.setFont("Helvetica", 10)
    line("Built by: Bara'u Magaji, Aliyu Muhammad Abdul, Aliyu Biniyaminu"); line("")

    p = lr.get('patient', {})
    line(f"Patient: {p.get('name','N/A')}    Age: {p.get('age','N/A')}    Sex: {p.get('sex','N/A')}")
    line(f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} "); line("")

    syms = lr.get('symptoms', [])
    line("Reported Symptoms:")
    for s in syms: line(f" • {s}")
    line("")

    r = lr.get('result', {})
    condition = r.get('condition', 'N/A')
    proba = r.get('proba')
    line(f"Likely Condition: {condition}")
    if proba is not None: line(f"Confidence: {proba:.0%}")
    line("")

    from textwrap import wrap
    advice = r.get('advice', '')
    precautions = r.get('precautions', [])
    line("Advice:")
    for chunk in wrap(advice, 90): line(chunk)
    if precautions:
        line(""); line("Precautions:")
        for pz in precautions: line(f" • {pz}")

    line(""); line("This guidance is informational and not a medical diagnosis.")
    c.showPage(); c.save(); buf.seek(0)

    filename = f"careguide_prescription_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
