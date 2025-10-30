# inference.py
import os
import io
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from flask import (
    Flask, request, jsonify, render_template_string, session, send_file, redirect, url_for
)
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash

# DB
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, Float, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ML + utils
import joblib
import numpy as np
from rapidfuzz import process, fuzz
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ---------------------------
# Config
# ---------------------------
APP = Flask(__name__)
CORS(APP)
APP.wsgi_app = ProxyFix(APP.wsgi_app)
APP.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
# Session (persistent on filesystem by default; use redis in prod via env)
APP.config['SESSION_TYPE'] = os.environ.get("SESSION_TYPE", "filesystem")
APP.config['SESSION_FILE_DIR'] = './flask_session'
APP.config['SESSION_PERMANENT'] = False
Session(APP)

# Files & env
MODEL_PKL = os.environ.get("MODEL_PKL", "model_tables.pkl")
SYMPTOM_EMB = os.environ.get("SYMPTOM_EMB", "symptom_embeddings.npz")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "change-this-admin-key")
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///careguide.db")
LOG_CSV = os.environ.get("LOG_CSV", "chat_logs.csv")

# ---------------------------
# Database (SQLAlchemy)
# ---------------------------
engine = create_engine(DB_URL, echo=False, future=True)
Base = declarative_base()
DBSession = sessionmaker(bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(256), unique=True, nullable=False)
    password_hash = Column(String(512), nullable=False)
    display_name = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    symptoms_entered = Column(Text)  # raw text
    matched_symptoms = Column(Text)  # canonical list joined by ;
    predicted_condition = Column(String(256))
    probability = Column(Float)
    advice = Column(Text)
    urgent_flag = Column(Boolean, default=False)
    external_sources = Column(Text)  # json dump
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="assessments")


Base.metadata.create_all(engine)

# ---------------------------
# Model loading
# ---------------------------
def safe_load_model():
    try:
        model_dict = joblib.load(MODEL_PKL)
    except Exception as e:
        print("Model load error:", e)
        model_dict = {}
    try:
        emb_npz = np.load(SYMPTOM_EMB, allow_pickle=True)
        emb = {k: emb_npz[k] for k in emb_npz.files}
    except Exception as e:
        print("Embeddings load error:", e)
        emb = {}
    return model_dict, emb


MODEL_DICT, SYM_EMB = safe_load_model()


class ModelBundle:
    def __init__(self, model_dict: Dict[str, Any], emb: Dict[str, Any]):
        self.clf = model_dict.get('clf')
        self.symptom_index = model_dict.get('symptom_index', {}) or {}
        self.disease_index = model_dict.get('disease_index', {}) or {}
        self.disease_info = model_dict.get('disease_info', {}) or {}
        self.vocab = list(emb.get('vocab', [])) if emb else []
        self.vectors = emb.get('vectors', None) if emb else None
        if isinstance(self.vectors, np.ndarray):
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self.vectors = self.vectors / norms

    def match_symptoms(self, raw_terms: List[str]) -> List[str]:
        # match via exact or embedding similarity fallback
        if not self.vectors or len(self.vocab) == 0:
            # exact only
            matched = [t for t in raw_terms if t in self.symptom_index]
            return matched
        matched = []
        for term in raw_terms:
            v = embed_text(term)
            if v is None:
                continue
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = self.vectors @ v
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= 0.45:
                matched.append(self.vocab[best_idx])
        # dedupe while preserving order
        seen = set()
        out = []
        for x in matched:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def predict(self, matched_symptoms: List[str]) -> Dict[str, Any]:
        x = np.zeros((1, len(self.symptom_index)), dtype=float)
        for s in matched_symptoms:
            idx = self.symptom_index.get(s)
            if idx is not None and idx < x.shape[1]:
                x[0, idx] = 1.0
        if self.clf is None:
            return {"condition": "model-missing", "proba": 0.0, "advice": "Model unavailable"}
        try:
            if hasattr(self.clf, "predict_proba"):
                proba_vec = self.clf.predict_proba(x)[0]
                top_idx = int(np.argmax(proba_vec))
                proba = float(proba_vec[top_idx])
            else:
                top_idx = int(self.clf.predict(x)[0])
                proba = None
        except Exception:
            top_idx = 0
            proba = None
        cond = self.decode_condition(top_idx)
        info = self.disease_info.get(cond, {})
        return {
            "condition": cond,
            "proba": proba,
            "advice": info.get("advice", "Follow up with your clinician"),
            "precautions": info.get("precautions", [])
        }

    def decode_condition(self, idx: int) -> str:
        if isinstance(self.disease_index, dict):
            return self.disease_index.get(idx, f"Class_{idx}")
        if isinstance(self.disease_index, (list, tuple)) and 0 <= idx < len(self.disease_index):
            return str(self.disease_index[idx])
        return f"Class_{idx}"


BUNDLE = ModelBundle(MODEL_DICT, SYM_EMB)

# ---------------------------
# Utils: embedding & fuzz
# ---------------------------
def embed_text(text: str, dim: int = 300) -> Optional[np.ndarray]:
    # deterministic pseudo-embedding (fast). Replace with a real model for production.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)


def fuzzy_suggest(term: str, limit: int = 3, cutoff: int = 60) -> List[str]:
    candidates = list(BUNDLE.symptom_index.keys())
    if not candidates:
        return []
    matches = process.extract(term, candidates, scorer=fuzz.WRatio, limit=limit, score_cutoff=cutoff)
    return [m[0] for m in matches]

# ---------------------------
# External medical APIs (with caching)
# ---------------------------
_external_cache = {}
CACHE_TTL = timedelta(hours=24)


def _cache_get(key: str):
    rec = _external_cache.get(key)
    if not rec:
        return None
    value, ts = rec
    if datetime.utcnow() - ts > CACHE_TTL:
        del _external_cache[key]
        return None
    return value


def _cache_set(key: str, value: Any):
    _external_cache[key] = (value, datetime.utcnow())


def fetch_healthfinder(condition: str) -> Dict[str, Any]:
    # simple cached fetch; real API might differ; keep safe
    key = f"hf:{condition}"
    c = _cache_get(key)
    if c is not None:
        return c
    try:
        url = "https://health.gov/myhealthfinder/api/v3/topicsearch.json"
        r = requests.get(url, params={"keyword": condition}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            _cache_set(key, data)
            return data
    except Exception:
        pass
    res = {"message": "No external healthfinder data"}
    _cache_set(key, res)
    return res


def fetch_openfda(drug: str) -> Dict[str, Any]:
    key = f"fda:{drug}"
    c = _cache_get(key)
    if c is not None:
        return c
    try:
        url = f"https://api.fda.gov/drug/label.json"
        params = {"search": f"openfda.brand_name:{drug}", "limit": 1}
        r = requests.get(url, params=params, timeout=6)
        if r.status_code == 200:
            _cache_set(key, r.json())
            return r.json()
    except Exception:
        pass
    out = {"message": "No openfda data"}
    _cache_set(key, out)
    return out


def fetch_who_lookup(term: str) -> Dict[str, Any]:
    key = f"who:{term}"
    c = _cache_get(key)
    if c is not None:
        return c
    # Placeholder: WHO APIs may require different endpoints
    res = {"message": "No WHO data (placeholder)"}
    _cache_set(key, res)
    return res

# ---------------------------
# Safety & urgent checks
# ---------------------------
URGENT_KEYWORDS = {
    "chest pain", "difficulty breathing", "shortness of breath", "severe bleeding",
    "loss of consciousness", "sudden weakness", "sudden numbness", "severe allergic reaction"
}


def check_urgent(symptoms: List[str]) -> bool:
    sset = set([t.lower() for t in symptoms])
    for k in URGENT_KEYWORDS:
        if k in sset:
            return True
    # also check tokens
    for term in symptoms:
        for k in URGENT_KEYWORDS:
            if k in term.lower():
                return True
    return False

# ---------------------------
# Session helpers & init
# ---------------------------
def init_state():
    # Only initialize if absent
    session.setdefault('stage', 'ask_symptoms')
    session.setdefault('symptoms', [])
    session.setdefault('pending_suggestions', [])
    session.setdefault('pending_term', None)
    session.setdefault('patient', {})
    session.setdefault('last_result', None)


def reset_state():
    session.clear()
    init_state()


# ---------------------------
# Embedded UI (blue-white theme)
# ---------------------------
HOME_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B.Magaji — HealthChero</title>
<style>
:root{--bg:#f8fbff;--card:#ffffff;--text:#0b2336;--accent:#0b84ff;--muted:#6b7280}
body{font-family:Inter,system-ui,Arial; margin:0; background:var(--bg); color:var(--text)}
.wrap{max-width:1000px;margin:24px auto;padding:20px}
.header{display:flex;align-items:center;gap:12px}
.logo{width:56px;height:56px;border-radius:8px;background:linear-gradient(135deg,#e6f0ff,#cfe6ff);display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--accent)}
.card{background:var(--card);border-radius:12px;padding:18px;box-shadow:0 6px 18px rgba(12,34,56,0.06)}
.controls{display:flex;gap:8px;margin-top:12px}
.chat{height:380px;overflow:auto;padding:12px;border-radius:8px;border:1px solid #e6eefb;background:#f7fbff}
.msg{margin:8px 0;display:block}
.ai{color:#052c4f;background:#eaf5ff;padding:8px;border-radius:8px;display:inline-block}
.user{color:#003150;text-align:right;display:block}
.row{display:flex;gap:8px;margin-top:12px}
input[type="text"], textarea{flex:1;padding:10px;border-radius:8px;border:1px solid #e6eefb}
button{background:var(--accent);color:white;padding:10px 12px;border-radius:8px;border:none;cursor:pointer}
.small{background:transparent;border:1px solid #e6eefb;color:var(--accent);padding:8px 10px;border-radius:8px}
.footer{margin-top:12px;color:var(--muted);font-size:13px}
.switch{margin-left:auto}
.badge{font-size:12px;background:#e6f0ff;color:var(--accent);padding:6px 8px;border-radius:999px}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="logo">BM</div>
    <div>
      <h2 style="margin:0">B.Magaji — HealthChero</h2>
      <div style="font-size:13px;color:var(--muted)">Built by Bara'u Magaji & team — Chat or use Quick Form</div>
    </div>
    <div class="switch">
      <button class="small" onclick="location.href='/guidelines'">Guidelines</button>
    </div>
  </div>

  <div style="height:16px"></div>

  <div class="card">
    <div style="display:flex;gap:12px;align-items:center">
      <div style="flex:1">
        <button onclick="setMode('chat')" id="btnChat" class="small">Chat</button>
        <button onclick="setMode('form')" id="btnForm" class="small">Quick Form</button>
      </div>
      <div style="display:flex;gap:8px">
        <button class="small" onclick="boot()">Start</button>
        <button id="btnRestart" class="small" onclick="restart()">Start Over</button>
      </div>
    </div>

    <div id="chatArea" style="margin-top:12px">
      <div class="chat" id="chat"></div>

      <div class="row" id="chatInputRow" style="margin-top:12px">
        <input id="chatInput" placeholder="Describe how you feel (e.g., fever, sore throat)" />
        <button onclick="sendChat()">Send</button>
      </div>
    </div>

    <div id="formArea" style="display:none;margin-top:12px">
      <form id="quickForm" onsubmit="submitForm(event)">
        <div style="display:flex;gap:8px">
          <input type="text" id="fname" placeholder="Patient name (optional)" />
          <input type="text" id="fage" placeholder="Age (optional)" style="width:120px"/>
          <input type="text" id="fsex" placeholder="Sex (male/female/other)" style="width:160px"/>
        </div>
        <div style="margin-top:8px">
          <textarea id="fsym" placeholder="Symptoms (comma separated)" style="width:100%;height:80px"></textarea>
        </div>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button type="submit">Submit Quick Form</button>
          <button type="button" class="small" onclick="setMode('chat')">Back to Chat</button>
        </div>
      </form>
    </div>

    <div class="footer">This tool provides informational guidance only — not a medical diagnosis. In emergencies call local emergency services.</div>
  </div>

  <div style="height:18px"></div>
  <div style="display:flex;gap:12px">
    <div class="card" style="flex:1">
      <h4 style="margin-top:0">Actions</h4>
      <div style="display:flex;gap:8px"><button onclick="downloadLast('simple')">Download Simple PDF</button><button onclick="downloadLast('professional')">Download Professional PDF</button></div>
      <div style="margin-top:8px"><button onclick="downloadCSV()">Admin: Download CSV</button></div>
    </div>
    <div class="card" style="width:320px">
      <h4 style="margin-top:0">Login (optional)</h4>
      <input id="li_email" placeholder="Email" style="width:100%;margin-bottom:6px"/>
      <input id="li_pass" placeholder="Password" type="password" style="width:100%;margin-bottom:6px"/>
      <div style="display:flex;gap:8px"><button onclick="login()">Login</button><button class="small" onclick="signup()">Sign up</button></div>
      <div id="authMsg" style="font-size:12px;color:#555;margin-top:6px"></div>
    </div>
  </div>

</div>

<script>
let mode = 'chat';
function setMode(m){
  mode = m;
  document.getElementById('chatArea').style.display = (m==='chat'?'block':'none');
  document.getElementById('formArea').style.display = (m==='form'?'block':'none');
  document.getElementById('btnChat').disabled = (m==='chat');
  document.getElementById('btnForm').disabled = (m==='form');
}

async function boot(){
  await fetch('/api/boot');
  appendAI("Hello! Describe your symptoms (e.g., fever, cough). Use 'done' when finished or use Quick Form.");
}
function appendAI(txt){
  const d=document.createElement('div'); d.className='msg ai'; d.textContent=txt; document.getElementById('chat').appendChild(d); document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
}
function appendUser(txt){
  const d=document.createElement('div'); d.className='msg user'; d.textContent=txt; document.getElementById('chat').appendChild(d); document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
}

async function sendChat(){
  const v = document.getElementById('chatInput').value.trim();
  if(!v) return;
  appendUser(v); document.getElementById('chatInput').value='';
  const r = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});
  const j = await r.json();
  appendAI(j.message);
}

async function submitForm(e){
  e.preventDefault();
  const name = document.getElementById('fname').value;
  const age = document.getElementById('fage').value;
  const sex = document.getElementById('fsex').value;
  const sym = document.getElementById('fsym').value;
  appendUser("Quick form submitted");
  const r = await fetch('/api/diagnose-form',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,age,sex,symptoms:sym})});
  const j = await r.json();
  appendAI(j.message);
}

async function restart(){
  await fetch('/api/restart',{method:'POST'});
  document.getElementById('chat').innerHTML='';
  boot();
}

async function login(){
  const email = document.getElementById('li_email').value;
  const pass = document.getElementById('li_pass').value;
  const r = await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email, password:pass})});
  const j = await r.json();
  document.getElementById('authMsg').textContent = j.message || JSON.stringify(j);
}

async function signup(){
  const email = document.getElementById('li_email').value;
  const pass = document.getElementById('li_pass').value;
  const r = await fetch('/api/signup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email, password:pass})});
  const j = await r.json();
  document.getElementById('authMsg').textContent = j.message || JSON.stringify(j);
}

async function downloadLast(style){
  window.location.href = `/download_pdf?style=${style}`;
}

async function downloadCSV(){
  const key = prompt("Admin key (required):");
  if(!key) return;
  window.location.href = `/download_csv?admin_key=${encodeURIComponent(key)}`;
}

boot();
</script>
</body>
</html>
"""

GUIDE_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Guidelines</title></head><body>
<h2>Clinical guidance (summaries)</h2>
<p>Important: This tool provides informational guidance only. For official clinical guidelines consult WHO, NHS, or local health authorities.</p>
<ul>
<li><a href="https://www.who.int">WHO</a></li>
<li><a href="https://www.nhs.uk">NHS</a></li>
</ul>
</body></html>
"""

# ---------------------------
# Routes: Auth & admin
# ---------------------------
@APP.route('/')
def home():
    init_state()
    return render_template_string(HOME_HTML)


@APP.route('/guidelines')
def guidelines():
    return render_template_string(GUIDE_HTML)


@APP.route('/api/signup', methods=['POST'])
def signup():
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({"ok": False, "message": "Email and password required"}), 400
    db = DBSession()
    if db.query(User).filter_by(email=email).first():
        return jsonify({"ok": False, "message": "Email already registered"}), 400
    user = User(email=email, password_hash=generate_password_hash(password), display_name=email.split('@')[0])
    db.add(user); db.commit()
    return jsonify({"ok": True, "message": "Account created. Please login."})


@APP.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    email = (data.get('email') or '').strip().lower(); password = data.get('password') or ''
    if not email or not password:
        return jsonify({"ok": False, "message": "Email and password required"}), 400
    db = DBSession()
    user = db.query(User).filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"ok": False, "message": "Invalid credentials"}), 401
    # store user id in session
    session['user_id'] = user.id
    session['user_email'] = user.email
    session['user_name'] = user.display_name
    return jsonify({"ok": True, "message": f"Logged in as {user.email}"})


@APP.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    session.pop('user_name', None)
    return jsonify({"ok": True})


# ---------------------------
# Conversation & form endpoints
# ---------------------------
@APP.route('/api/boot')
def api_boot():
    # initialize conversation once
    reset_state()
    session['stage'] = 'ask_symptoms'
    return jsonify({"message": "Hello! I'm HealthChero (B.Magaji). Describe your symptoms separated by commas, or use the Quick Form."})


@APP.route('/api/restart', methods=['POST'])
def api_restart():
    reset_state()
    return jsonify({"ok": True})


def parse_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]


@APP.route('/api/message', methods=['POST'])
def api_message():
    init_state()
    data = request.json or {}
    text = (data.get('message') or '').strip()
    stage = session.get('stage', 'ask_symptoms')

    # handle pending suggestions (user chose 1/2/3 or typed yes/no)
    if session.get('pending_suggestions'):
        suggestions = session.get('pending_suggestions') or []
        low = text.lower().strip()
        if low in ('skip', 'no'):
            session['pending_suggestions'] = []
            session['pending_term'] = None
            return jsonify({"message": "Okay, skipped that term. Add more symptoms or type 'done'."})
        # numeric choices
        if low in ('1', '2', '3'):
            idx = int(low) - 1
            if 0 <= idx < len(suggestions):
                chosen = suggestions[idx]
                syms = session.get('symptoms', [])
                if chosen not in syms:
                    syms.append(chosen)
                session['symptoms'] = syms
                session['pending_suggestions'] = []
                session['pending_term'] = None
                return jsonify({"message": f"Added: {chosen}. Add more or type 'done'."})
        # typed exact suggestion
        for s in suggestions:
            if low == s.lower():
                syms = session.get('symptoms', [])
                if s not in syms:
                    syms.append(s)
                session['symptoms'] = syms
                session['pending_suggestions'] = []
                session['pending_term'] = None
                return jsonify({"message": f"Added: {s}. Add more or type 'done'."})
        return jsonify({"message": f"Please reply 1/2/3, type the correct word, or 'skip'. Suggestions: {', '.join([f'{i+1}. {w}' for i,w in enumerate(suggestions)])}"})

    # main state machine
    if stage == 'ask_symptoms':
        if text.lower() in ('done', 'finish', 'end'):
            if not session.get('symptoms'):
                return jsonify({"message": "Please add at least one symptom before finishing."})
            session['stage'] = 'ask_name'
            return jsonify({"message": "Got it. Please enter patient name (optional) or type 'skip'."})
        tokens = parse_tokens(text)
        if not tokens:
            return jsonify({"message": "Please describe your symptoms (comma separated) or use Quick Form."})
        added = []
        for term in tokens:
            if term in BUNDLE.symptom_index:
                if term not in session['symptoms']:
                    session['symptoms'].append(term); added.append(term)
            else:
                suggestions = fuzzy_suggest(term, limit=3, cutoff=60)
                if suggestions:
                    session['pending_suggestions'] = suggestions
                    session['pending_term'] = term
                    s_list = ", ".join([f"{i+1}. {w}" for i, w in enumerate(suggestions)])
                    return jsonify({"message": f"I didn't recognize '{term}'. Did you mean: {s_list}? Reply 1/2/3, type the correct word, or 'skip'."})
                # otherwise ignore unknown token quietly
        if added:
            return jsonify({"message": f"Symptoms noted: {', '.join(added)}. Add more or type 'done'."})
        return jsonify({"message": "Thanks. Add more symptoms or type 'done' to proceed."})

    if stage == 'ask_name':
        if text.lower() not in ('skip','none','n/a',''):
            # only save name in session (not stored in DB per your Q2=A)
            session['patient']['name'] = text
        session['stage'] = 'ask_age'
        return jsonify({"message": "Age? (optional) or type 'skip'."})

    if stage == 'ask_age':
        if text.lower() not in ('skip','none','n/a',''):
            digits = ''.join(ch for ch in text if ch.isdigit())
            if digits:
                session['patient']['age'] = int(digits)
        session['stage'] = 'ask_sex'
        return jsonify({"message": "Sex (male/female/other) or type 'skip'."})

    if stage == 'ask_sex':
        if text.lower() not in ('skip','none','n/a',''):
            session['patient']['sex'] = text.lower()
        # now compute
        symptoms_entered = session.get('symptoms', [])[:]
        matched = BUNDLE.match_symptoms(symptoms_entered)
        result = BUNDLE.predict(matched)
        urgent = check_urgent(matched + symptoms_entered)
        # save minimal assessment in DB (Q2=A)
        db = DBSession()
        ass = Assessment(
            user_id=session.get('user_id'),
            symptoms_entered="; ".join(symptoms_entered),
            matched_symptoms="; ".join(matched),
            predicted_condition=result.get('condition'),
            probability=float(result.get('proba') or 0.0),
            advice=result.get('advice'),
            urgent_flag=bool(urgent),
            external_sources=json.dumps({
                "healthfinder": fetch_healthfinder(result.get('condition')),
                "who": fetch_who_lookup(result.get('condition'))
            })
        )
        db.add(ass); db.commit()
        session['last_result'] = {
            "assessment_id": ass.id,
            "result": result,
            "matched": matched
        }
        session['stage'] = 'done'
        proba_txt = f" (confidence {result['proba']:.0%})" if result.get('proba') else ""
        extras = ""
        if result.get('precautions'):
            extras = "\nPrecautions: " + ", ".join(result.get('precautions'))
        urgent_msg = ""
        if urgent:
            urgent_msg = "\n\n⚠️ URGENT: Your symptoms include red-flag items. Seek emergency care immediately if severe."
        return jsonify({"message": f"Based on your symptoms, possible: {result.get('condition')}{proba_txt}.\nAdvice: {result.get('advice')}{extras}{urgent_msg}\nYou can download a PDF or type 'start over'."})

    # completed
    return jsonify({"message": "Session complete. Click Start Over to run a new assessment."})


# Quick form handler that mirrors form-based flow
@APP.route('/api/diagnose-form', methods=['POST'])
def api_diagnose_form():
    init_state()
    data = request.json or {}
    name = data.get('name') or ''
    age = data.get('age') or ''
    sex = data.get('sex') or ''
    symptoms_raw = data.get('symptoms') or ''
    tokens = parse_tokens(symptoms_raw)
    session['symptoms'] = tokens
    session['patient'] = {}
    if name:
        session['patient']['name'] = name
    if age:
        digits = ''.join(ch for ch in str(age) if ch.isdigit())
        if digits:
            session['patient']['age'] = int(digits)
    if sex:
        session['patient']['sex'] = sex.lower()
    # perform prediction like above
    matched = BUNDLE.match_symptoms(tokens)
    result = BUNDLE.predict(matched)
    urgent = check_urgent(matched + tokens)
    db = DBSession()
    ass = Assessment(
        user_id=session.get('user_id'),
        symptoms_entered="; ".join(tokens),
        matched_symptoms="; ".join(matched),
        predicted_condition=result.get('condition'),
        probability=float(result.get('proba') or 0.0),
        advice=result.get('advice'),
        urgent_flag=bool(urgent),
        external_sources=json.dumps({
            "healthfinder": fetch_healthfinder(result.get('condition')),
            "who": fetch_who_lookup(result.get('condition'))
        })
    )
    db.add(ass); db.commit()
    session['last_result'] = {"assessment_id": ass.id, "result": result, "matched": matched}
    session['stage'] = 'done'
    proba_txt = f" (confidence {result['proba']:.0%})" if result.get('proba') else ""
    urgent_msg = "\n\n⚠️ URGENT: Seek emergency care if symptoms severe." if urgent else ""
    return jsonify({"message": f"Quick result: {result.get('condition')}{proba_txt}. Advice: {result.get('advice')}{urgent_msg}"})


# ---------------------------
# Downloads & admin
# ---------------------------
def make_pdf_simple(assessment: Assessment):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20*mm
    def line(t, step=7):
        nonlocal y
        c.setFont("Helvetica", 10); c.drawString(20*mm, y, t); y -= step*mm
    line("HealthChero — Simple Report")
    line(f"Assessment ID: {assessment.id}")
    line(f"Date: {assessment.created_at.isoformat()}")
    line("")
    line("Symptoms Entered:")
    for s in assessment.symptoms_entered.split(";"):
        line(f" - {s.strip()}")
    line("")
    line(f"Predicted: {assessment.predicted_condition} (prob: {assessment.probability:.2f})")
    line("Advice:")
    for seg in (assessment.advice or "").split("\n"):
        line(seg)
    c.showPage(); c.save(); buf.seek(0)
    return buf


def make_pdf_professional(assessment: Assessment, patient_info: Dict[str, Any]):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20*mm
    def line(t, step=6):
        nonlocal y
        c.setFont("Helvetica", 11); c.drawString(20*mm, y, t); y -= step*mm
    c.setFont("Helvetica-Bold", 16); line("HealthChero — Professional Medical Report", step=10)
    c.setFont("Helvetica", 10)
    line(f"Assessment ID: {assessment.id}")
    line(f"Date: {assessment.created_at.isoformat()}")
    line(f"Patient: {patient_info.get('name','Anonymous')}  Age: {patient_info.get('age','N/A')}  Sex: {patient_info.get('sex','N/A')}")
    line("")
    line("Symptoms Entered:")
    for s in assessment.symptoms_entered.split(";"):
        line(f" • {s.strip()}")
    line("")
    line(f"Predicted Condition: {assessment.predicted_condition} (prob: {assessment.probability:.2f})")
    line("")
    line("Advice and Precautions:")
    for seg in (assessment.advice or "").split("\n"):
        for chunk in [seg[i:i+100] for i in range(0, len(seg), 100)]:
            line(chunk)
    c.showPage(); c.save(); buf.seek(0)
    return buf


@APP.route('/download_pdf')
def download_pdf():
    last = session.get('last_result')
    if not last:
        return "No assessment available", 400
    aid = last.get('assessment_id')
    style = (request.args.get('style') or 'simple').lower()
    db = DBSession()
    ass = db.query(Assessment).filter_by(id=aid).first()
    if not ass:
        return "Assessment not found", 404
    patient_info = session.get('patient', {})
    if style == 'professional':
        buf = make_pdf_professional(ass, patient_info)
        fname = f"healthchero_report_{ass.id}_professional.pdf"
    else:
        buf = make_pdf_simple(ass)
        fname = f"healthchero_report_{ass.id}_simple.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/pdf")


@APP.route('/download_csv')
def download_csv():
    key = request.args.get('admin_key') or ''
    if not key or key != ADMIN_KEY:
        return "Unauthorized", 403
    # produce CSV of assessments
    db = DBSession()
    rows = db.query(Assessment).order_by(Assessment.created_at.desc()).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id","created_at","symptoms_entered","matched_symptoms","predicted_condition","probability","urgent_flag"])
    for r in rows:
        writer.writerow([r.id, r.created_at.isoformat(), r.symptoms_entered, r.matched_symptoms, r.predicted_condition, r.probability, r.urgent_flag])
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='assessments.csv')


# ---------------------------
# Health endpoint
# ---------------------------
@APP.route('/health')
def health():
    return jsonify({"status":"ok","model_loaded": bool(MODEL_DICT), "embeddings_loaded": bool(SYM_EMB)})


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
