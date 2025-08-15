import os, io
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import joblib
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, url_for
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from rapidfuzz import process

# =============================
# App setup
# =============================
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-please")

ARTIFACT_MODEL = os.environ.get("MODEL_PKL", "model_tables.pkl")
ARTIFACT_EMBED = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")

# =============================
# Model bundle
# =============================
class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], symptom_embeddings: Dict[str, Any]):
        self.clf = model_dict.get('clf', None)
        self.symptom_index = model_dict.get('symptom_index', {}) or {}
        self.disease_index = model_dict.get('disease_index', {}) or {}
        self.disease_info = model_dict.get('disease_info', {}) or {}

        # Optional embedding helper (not strictly needed for inference)
        self.vocab = list(symptom_embeddings.get('vocab', [])) if symptom_embeddings else []
        self.vectors = symptom_embeddings.get('vectors', None) if symptom_embeddings else None
        if isinstance(self.vectors, np.ndarray) and self.vectors.size > 0:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        if self.clf is None or not self.symptom_index:
            return {
                "condition": "Model not loaded",
                "proba": None,
                "matched_symptoms": [],
                "advice": "Model artifacts missing. Please retrain and redeploy.",
                "precautions": []
            }

        # One-hot encode over known symptoms
        x = np.zeros((1, len(self.symptom_index)), dtype=float)
        for s in symptoms:
            idx = self.symptom_index.get(s)
            if idx is not None:
                x[0, idx] = 1.0

        # Predict
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
        advice = info.get('advice', 'Rest, stay hydrated, and seek medical help if symptoms worsen.')
        precautions = info.get('precautions', ['Stay hydrated', 'Rest well'])

        return {
            "condition": condition,
            "proba": proba,
            "matched_symptoms": symptoms,
            "advice": advice,
            "precautions": precautions
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            # disease_index: {int -> name}
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"


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

# =============================
# Conversation state
# =============================
def init_state():
    session.setdefault('stage', 'greet')
    session.setdefault('patient', {})
    session.setdefault('symptoms', [])
    session.setdefault('last_result', None)
    session.setdefault('pending_suggestions', [])   # holds suggestions for a single unknown term
    session.setdefault('pending_term', None)

def reset_state():
    session.clear()
    init_state()

# =============================
# HTML (keeps your style/colors)
# =============================
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
    .attribution{font-size:14px; opacity:.85; margin-bottom:12px}
    .chat{height:50vh; overflow:auto; border:1px solid #1f2937; border-radius:12px; padding:12px; background:#0b1220}
    .msg{margin:8px 0; white-space:pre-wrap}
    .ai{color:#a7f3d0}
    .user{color:#93c5fd; text-align:right}
    .row{display:flex; gap:8px; margin-top:12px}
    input{flex:1; background:#0b1220; border:1px solid #1f2937; color:#e5e7eb; padding:10px; border-radius:10px}
    button{background:#22c55e; color:#052e16; border:none; padding:10px 14px; border-radius:10px; font-weight:700; cursor:pointer}
    .download{background:#38bdf8; color:#06263a}
    .bar{display:flex; gap:10px; align-items:center; margin-bottom:10px}
    .link{color:#93c5fd; text-decoration:underline; cursor:pointer}
    .footer{margin-top:10px; font-size:12px; opacity:.7}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="bar">
        <h1 style="flex:1">CareGuide AI</h1>
        <a class="link" href="/guidelines" target="_blank">Guidelines</a>
        <a class="link" href="/health">Health</a>
      </div>
      <div class="attribution">Built by <strong>Bara'u Magaji</strong>, <strong>Aliyu Muhammad Abdul</strong>, and <strong>Aliyu Biniyaminu</strong>.</div>
      <div class="chat" id="chat"></div>
      <div class="row">
        <input id="input" placeholder="Type your reply here..." />
        <button onclick="send()">Send</button>
      </div>
      <div class="row">
        <button class="download" onclick="downloadPdf()">Download last prescription (PDF)</button>
        <button onclick="restart()">Start over</button>
      </div>
      <div class="footer">CareGuide AI provides educational guidance and is not a substitute for professional medical advice.</div>
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
  append('ai', j.message);
}

async function restart(){
  await fetch('/api/restart', { method:'POST' });
  chat.innerHTML = '';
  boot();
}

async function downloadPdf(){
  const r = await fetch('/api/has_result');
  const j = await r.json();
  if(!j.ok){
    append('ai', 'No prescription available yet. Please complete an assessment first.');
    return;
  }
  window.location.href = '/download_pdf';
}

boot();
</script>
</body>
</html>
"""

GUIDE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CareGuide AI — Guidelines</title>
  <style>
    body{font-family: system-ui; margin:0; background:#0f172a; color:#e2e8f0}
    .wrap{max-width:900px; margin:0 auto; padding:24px}
    .card{background:#111827; border-radius:16px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.25)}
    h1{margin:0 0 12px 0}
    h2{margin:16px 0 8px 0}
    p, li{line-height:1.5}
    a{color:#93c5fd}
    code{background:#0b1220; padding:2px 6px; border-radius:6px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>How to use CareGuide AI</h1>
      <p><strong>What it is:</strong> an educational assistant that provides general guidance based on symptoms you describe. It is <em>not</em> a medical diagnosis and does not replace a clinician.</p>

      <h2>Quick start</h2>
      <ol>
        <li>On the home page, type your symptoms separated by commas (e.g., <code>fever, sore throat, cough</code>) and press <em>Send</em>.</li>
        <li>If a term isn’t recognized, you’ll get up to three suggestions. Reply with the correct word or type <code>skip</code> to ignore.</li>
        <li>Type <code>done</code> when you’ve finished listing symptoms.</li>
        <li>Answer a few demographics (name, age, sex). Then you’ll receive a likely condition, advice, and precautions.</li>
        <li>Click <em>Download last prescription (PDF)</em> for a summary.</li>
      </ol>

      <h2>Tips for better results</h2>
      <ul>
        <li>Use simple words for symptoms (e.g., “nausea”, “vomiting”, “abdominal pain”).</li>
        <li>Add duration and severity in plain text (e.g., “fever for 3 days”).</li>
        <li>Provide related signs (e.g., “rash”, “shortness of breath”, “chest pain”).</li>
      </ul>

      <h2>Safety</h2>
      <ul>
        <li>Seek emergency care for red flags: severe chest pain, difficulty breathing, confusion, fainting, signs of stroke, heavy bleeding, or severe allergic reaction.</li>
        <li>This tool may be inaccurate—always consult a healthcare professional.</li>
      </ul>

      <p><a href="/">← Back to app</a></p>
    </div>
  </div>
</body>
</html>
"""

# =============================
# Routes
# =============================
@app.route('/')
def home():
    init_state()
    return render_template_string(HOME_HTML)

@app.route('/guidelines')
def guidelines():
    return render_template_string(GUIDE_HTML)

@app.route('/api/boot')
def api_boot():
    init_state()
    session['stage'] = 'ask_symptoms'
    msg = (
        "Hello! I'm CareGuide AI.\n"
        "Built by Bara'u Magaji, Aliyu Muhammad Abdul, and Aliyu Biniyaminu.\n\n"
        "Describe your symptoms separated by commas (e.g., 'fever, cough'). "
        "Type 'done' when finished."
    )
    return jsonify({"message": msg})

@app.route('/api/restart', methods=['POST'])
def api_restart():
    reset_state()
    return jsonify({"ok": True})

# ---------- helpers ----------
def parse_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]

def suggest_terms(term: str) -> List[str]:
    candidates = list(BUNDLE.symptom_index.keys())
    if not candidates:
        return []
    # top-3 suggestions if score >= 65
    matches = process.extract(term, candidates, limit=3, score_cutoff=65)
    return [m[0] for m in matches]

# ---------- conversation ----------
@app.route('/api/message', methods=['POST'])
def api_message():
    init_state()
    data = request.get_json(force=True)
    text = (data.get('message') or '').strip()

    stage = session.get('stage', 'ask_symptoms')

    # Handle a suggestion confirmation step
    if session.get('pending_suggestions'):
        sug = session['pending_suggestions']
        # user can answer with index "1/2/3", exact word, or "skip"
        low = text.lower()
        chosen = None
        if low in ('skip', 'no'):
            # ignore the unknown term and continue collecting
            session['pending_suggestions'] = []
            session['pending_term'] = None
            return jsonify({"message": "Okay, skipped. Add more symptoms or type 'done' when you're finished."})
        if low in ('1','2','3'):
            idx = int(low) - 1
            if 0 <= idx < len(sug):
                chosen = sug[idx]
        elif low in [s.lower() for s in sug]:
            chosen = low
        if chosen:
            syms = session.get('symptoms', [])
            if chosen not in syms:
                syms.append(chosen)
            session['symptoms'] = syms
            session['pending_suggestions'] = []
            session['pending_term'] = None
            return jsonify({"message": f"Added: {chosen}. Add more symptoms or type 'done'."})
        # If reply didn’t match, remind the user
        return jsonify({"message": f"Please reply 1, 2, or 3; type the correct word; or 'skip'. Suggestions: {', '.join(f'{i+1}. {w}' for i,w in enumerate(sug))}"})

    if stage == 'ask_symptoms':
        # If user finished
        if text.lower() in ('done', 'finish', 'end', 'no'):
            if not session.get('symptoms'):
                return jsonify({"message": "Please add at least one symptom before finishing."})
            session['stage'] = 'ask_name'
            return jsonify({"message": "Got it. What is your full name?"})

        tokens = parse_tokens(text)
        accepted = []
        for term in tokens:
            if term in BUNDLE.symptom_index:
                accepted.append(term)
            else:
                suggestions = suggest_terms(term)
                if suggestions:
                    session['pending_suggestions'] = suggestions
                    session['pending_term'] = term
                    s_list = ", ".join([f"{i+1}. {w}" for i, w in enumerate(suggestions)])
                    return jsonify({"message": f"I didn’t recognize '{term}'. Did you mean: {s_list}? Reply 1/2/3, type the correct word, or 'skip'."})
                # no suggestion – ignore quietly
        # Save any accepted
        if accepted:
            syms = session.get('symptoms', [])
            for a in accepted:
                if a not in syms:
                    syms.append(a)
            session['symptoms'] = syms
            return jsonify({"message": f"Symptoms noted: {', '.join(accepted)}. Add more or type 'done'."})
        return jsonify({"message": "Thanks. Add more symptoms or type 'done' to proceed."})

    if stage == 'ask_name':
        session.setdefault('patient', {})['name'] = text
        session['stage'] = 'ask_age'
        return jsonify({"message": f"Thanks {text}. How old are you?"})

    if stage == 'ask_age':
        age_num = ''.join(ch for ch in text if ch.isdigit())
        session['patient']['age'] = int(age_num) if age_num.isdigit() else None
        session['stage'] = 'ask_sex'
        return jsonify({"message": "What is your sex? (male/female/other)"})

    if stage == 'ask_sex':
        session['patient']['sex'] = text.lower()
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
        precautions = result.get('precautions', [])
        extra = ("\nPrecautions: " + ", ".join(precautions)) if precautions else ""
        return jsonify({"message":
            f"Based on your symptoms, you may be experiencing: {cond}{proba_txt}.\n\n"
            f"Advice: {result['advice']}{extra}\n\n"
            f"You can download a PDF summary or click Start over."
        })

    # Fallback
    return jsonify({"message": "Session completed. Click Start over to begin a new assessment."})

# =============================
# Utility routes
# =============================
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

    c.setTitle("CareGuide AI — Summary")
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

    line(""); line("Disclaimer: Informational only, not a medical diagnosis.")
    c.showPage(); c.save(); buf.seek(0)

    filename = f"careguide_prescription_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# =============================
# Main
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
