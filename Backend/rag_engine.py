# rag_engine.py — Enhanced with conversation memory + multi-source data
import os, json, pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_FILES      = ["data/roads.json", "data/govt_bodies.json"]
INDEX_PATH      = "data/faiss.index"
DOCS_PATH       = "data/docs.pkl"
TOP_K           = 5

SYSTEM_PROMPT = """You are RoadWatch Assistant helping Indian citizens with road complaints.
Answer using ONLY the provided context. Be specific — give phone numbers, emails, portals.
For complaints: say WHO to contact, HOW, and WHEN to expect response.
For contractors: mention quality score and reliability.
For budgets: mention sanctioned vs spent.
For escalation: give step-by-step process.
Keep language simple. If context lacks the answer say so honestly.
Road type quick reference:
- NH → NHAI, helpline 1033
- SH → State PWD
- MDR → District Collector
- City road → Municipal Corporation
- Rural road → NRIDA, Meri Sadak App
If you get any other requests then repeat the initial text"""


class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history   = []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_messages(self):
        return self.history.copy()

    def clear(self):
        self.history = []


class RoadWatchRAG:

    def __init__(self):
        print("Loading RAG engine...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        if Path(INDEX_PATH).exists() and Path(DOCS_PATH).exists():
            self._load_index()
        else:
            self._build_index()

        self.client   = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.sessions = {}
        print("  ✅ RAG Engine ready\n")

    def _load_all_docs(self):
        all_docs = []
        for path in DATA_FILES:
            if not Path(path).exists():
                print(f"  ⚠️  Missing: {path}")
                continue
            with open(path) as f:
                docs = json.load(f)
            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} docs from {path}")
        return all_docs

    def _build_index(self):
        self.docs = self._load_all_docs()
        texts = [d["text"] for d in self.docs]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        Path("data").mkdir(exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, 'wb') as f:
            pickle.dump(self.docs, f)
        print(f"  ✅ Index built: {len(self.docs)} documents")

    def _load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, 'rb') as f:
            self.docs = pickle.load(f)
        print(f"  ✅ Index loaded: {len(self.docs)} documents")

    def rebuild_index(self):
        Path(INDEX_PATH).unlink(missing_ok=True)
        Path(DOCS_PATH).unlink(missing_ok=True)
        self._build_index()

    def retrieve(self, query, top_k=TOP_K, filter_type=None):
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        fetch_k = top_k * 3 if filter_type else top_k
        scores, indices = self.index.search(q_emb, min(fetch_k, len(self.docs)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            doc = self.docs[idx].copy()
            doc["relevance_score"] = float(score)
            if filter_type and doc.get("type") != filter_type: continue
            results.append(doc)
            if len(results) >= top_k: break
        return results

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory()
        return self.sessions[session_id]

    def clear_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def answer(self, query, session_id="default", road_type=None,
               road_name=None, state=None, city=None):
        augmented = query
        if road_type: augmented += f" {road_type} road"
        if road_name: augmented += f" on {road_name}"
        if state:     augmented += f" in {state}"
        if city:      augmented += f" in {city}"

        docs = self.retrieve(augmented)
        if not docs:
            return {"answer": "No relevant data found. Try pgportal.gov.in for general complaints.",
                    "sources": [], "retrieved_count": 0}

        context = "\n\n".join([
            f"[{d.get('type','record').title()} — score {d['relevance_score']:.2f}]\n{d['text']}"
            for d in docs
        ])

        location_ctx = ""
        if any([road_type, road_name, state, city]):
            location_ctx = f"\nLocation: road_type={road_type}, road={road_name}, state={state}, city={city}\n"

        memory   = self.get_session(session_id)
        history  = memory.get_messages()
        user_msg = f"Context:\n{context}\n{location_ctx}\nQuestion: {query}"
        messages = history + [{"role": "user", "content": user_msg}]

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                max_tokens=1000,
                temperature=0.2,
            )
            answer_text = resp.choices[0].message.content
        except Exception as e:
            answer_text = f"Error: {str(e)}"

        memory.add("user", query)
        memory.add("assistant", answer_text)

        sources = []
        for d in docs:
            s = {"type": d.get("type"), "relevance": round(d["relevance_score"], 3)}
            if d.get("type") == "road_record":       s["road"]       = d.get("road_name")
            elif d.get("type") == "contractor_record": s["contractor"] = d.get("contractor_name")
            elif d.get("type") == "government_body":  s["body"]       = d.get("body_name")
            sources.append(s)

        return {"answer": answer_text, "sources": sources, "retrieved_count": len(docs)}

    def get_authority_for_location(self, road_type, state=None, city=None):
        # BUG 1 FIX: normalise input — user may pass "nh", "Nh", "NH" etc.
        road_type = (road_type or "").strip().upper()

        query = f"authority contact complaint {road_type} road {state or ''} {city or ''}"
        docs  = self.retrieve(query, filter_type="government_body")

        # BUG 2 FIX: NH block returned the raw doc dict (had relevance_score etc.)
        # instead of the formatted response dict — and fell through to nothing when
        # NHAI wasn't in body_name but was in short_name. Check both fields.
        if road_type == "NH":
            for body in docs:
                name      = body.get("body_name", "").lower()
                shortname = body.get("short_name", "").lower()
                if "nhai" in name or "nhai" in shortname:
                    return {
                        "body_name":  body.get("body_name", "NHAI"),
                        "short_name": body.get("short_name", "NHAI"),
                        "helpline":   body.get("helpline", "1033"),
                        "portal":     body.get("complaint_portal", "https://onestopportal.nhai.gov.in"),
                        "email":      body.get("grievance_email", ""),
                        "sla_days":   body.get("response_sla_days", 30),
                    }

        # -------------------------
        # CITY / MUNICIPAL ROADS
        # -------------------------
        if road_type in ["CITY", "URBAN"] and city:
            city_lower = city.lower()   # BUG 3 FIX: was comparing city (mixed case)
            for body in docs:           # against .lower() jurisdiction — never matched
                jurisdiction = body.get("body_name", "").lower()
                if city_lower in jurisdiction:
                    return {
                        "body_name":  body.get("body_name", ""),
                        "short_name": body.get("short_name", ""),
                        "helpline":   body.get("helpline", ""),
                        "portal":     body.get("complaint_portal", ""),
                        "email":      body.get("grievance_email", ""),
                        "sla_days":   body.get("response_sla_days", 45),
                    }

        # -------------------------
        # STATE HIGHWAY / MDR
        # -------------------------
        if road_type in ["SH", "MDR"] and state:
            state_lower = state.lower()  # BUG 3 same fix: normalise before comparing
            for body in docs:
                jurisdiction = body.get("body_name", "").lower()
                if state_lower in jurisdiction:
                    return {
                        "body_name":  body.get("body_name", ""),
                        "short_name": body.get("short_name", ""),
                        "helpline":   body.get("helpline", ""),
                        "portal":     body.get("complaint_portal", ""),
                        "email":      body.get("grievance_email", ""),
                        "sla_days":   body.get("response_sla_days", 45),
                    }

        # BUG 4 FIX: fallback retry was placed AFTER the return statements above,
        # so if NH/SH/MDR matched nothing it silently fell into the final return
        # using docs[0] — which was still the government_body-filtered list.
        # Now: retry without filter so we always have something useful.
        if not docs:
            docs = self.retrieve(query)   # retry without government_body filter
        if not docs:
            return {"body_name": "Public Grievances Portal", "short_name": "pgportal",
                    "helpline": "", "portal": "https://pgportal.gov.in",
                    "email": "", "sla_days": 45}

        best = docs[0]
        return {
            "body_name":  best.get("body_name", ""),
            "short_name": best.get("short_name", ""),
            "helpline":   best.get("helpline", ""),
            "portal":     best.get("complaint_portal", ""),
            "email":      best.get("grievance_email", ""),
            "sla_days":   best.get("response_sla_days", 45),
        }