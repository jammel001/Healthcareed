
import pickle, json, math, numpy as np
from sentence_transformers import SentenceTransformer, util

class HealthInference:
    def __init__(self, tables_path="health_model/model_tables.pkl", emb_path="health_model/symptom_embeddings.npz"):
        with open(tables_path, "rb") as f:
            data = pickle.load(f)
        self.symptom_vocab = data["symptom_vocab"]
        self.priors = data["priors"]
        self.cond = data["cond"]
        self.disease_info = data.get("disease_info", {})
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        npz = np.load(emb_path, allow_pickle=True)
        self.embeddings = npz["embeddings"]

    def rank(self, symptoms, top_k=5):
        symptoms = [s.lower().strip() for s in symptoms if s]
        diseases = list(self.priors.keys())
        rows = []
        for d in diseases:
            lp = math.log(self.priors.get(d, 1e-12))
            for s in symptoms:
                lp += math.log(self.cond[d].get(s, 1e-12))
            rows.append((d, lp))
        rows.sort(key=lambda x: x[1], reverse=True)
        out = rows[:top_k]
        results = []
        maxlp = out[0][1]
        for d, lp in out:
            score = math.exp(lp - maxlp)
            info = self.disease_info.get(d, {})
            results.append({"disease": d, "score": score, **info})
        return results

    def nearest(self, query, top_k=3):
        q = [query.lower().strip()]
        qemb = self.embedder.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        sim = util.cos_sim(qemb, self.embeddings).cpu().numpy()[0]
        idx = np.argsort(sim)[::-1][:top_k]
        return [(self.symptom_vocab[i], float(sim[i])) for i in idx]
