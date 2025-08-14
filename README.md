# CareGuide AI

Conversational health helper that:
- Greets patient + shows: Bara'u Magaji, Aliyu Muhammad Abdul, Aliyu Biniyaminu
- Asks for **feelings first**, then collects **name → age → gender**
- Predicts likely condition + advice
- Offers a PDF “prescription” download

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_SECRET_KEY="some-long-random"
python inference.py
# open http://localhost:10000
