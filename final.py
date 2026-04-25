# =============================================================================
#  WEEK 3 — AI CUSTOMER SUPPORT COPILOT  (Gradio + Groq/Gemini)
#  Fixed version — all demo issues resolved
#
#  FIXES IN THIS VERSION:
#  1. Confidence always low  → LinearSVC softmax scaled properly (150 classes)
#  2. Filter always "none"   → thresholds lowered to match real confidence range
#  3. Button label= error    → removed keyword argument
#  4. Broken avatars         → SVG data-URIs (no external dependency)
#  5. Dark/Light theme       → toggle in header
#  6. Model switcher         → Groq (5 models) + Gemini Flash dropdown
#  7. Out-of-scope query     → added to sample queries, guardrail shown
#  8. Eval chart removed     → cleaned up
#  9. Week 2 retrieval summary removed → cleaned up
#
#  INSTALL:
#  pip install gradio groq google-generativeai sentence-transformers
#              faiss-cpu chromadb rouge-score scikit-learn -q
#
#  RUN:  python week3_app_gradio.py
# =============================================================================

import os, re, pickle, time, logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import gradio as gr
from groq import Groq

from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from rouge_score import rouge_scorer

# ── Suppress noisy warnings ───────────────────────────────────────────────
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# =============================================================================
#  CONFIGURATION
# =============================================================================

ARTIFACTS_DIR = "artifacts"

# ── Available models ──────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.1-8b-instant  (fastest, recommended)":  "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile (most capable)":        "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile (balanced)":            "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768 (large context)":            "mixtral-8x7b-32768",
    "gemma2-9b-it (Google/Groq)":                    "gemma2-9b-it",
}
GEMINI_MODELS = {
    "gemini-2.0-flash (free, fast)":  "gemini-2.0-flash",
    "gemini-1.5-flash (free, stable)": "gemini-1.5-flash",
}
ALL_MODEL_LABELS = list(GROQ_MODELS.keys()) + list(GEMINI_MODELS.keys())
DEFAULT_MODEL_LABEL = "llama-3.1-8b-instant  (fastest, recommended)"

# ── Pipeline thresholds ───────────────────────────────────────────────────
# WHY THESE VALUES:
# LinearSVC with 150 classes produces decision_function scores that, after
# softmax over 150 classes, rarely exceed 0.15 even for correct predictions.
# The raw max softmax value with 150 classes averages ~1/150 ≈ 0.007 for
# uniform distribution. A correct prediction sits at ~0.05–0.20.
# We scale confidence to [0,1] using rank-based rescaling so the thresholds
# make intuitive sense. See predict_intent() for details.
CONF_HIGH = 0.55   # above → intent-level filter
CONF_LOW  = 0.25   # above → domain-level filter, below → no filter

TOP_K         = 3
MAX_INPUT_LEN = 500
MIN_INPUT_LEN = 3
MAX_HISTORY   = 10
DRIFT_WINDOW  = 3

# ── SVG avatars as data-URIs (no external dependency, works offline) ──────
USER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'>"
    "<circle cx='20' cy='20' r='20' fill='%234A90D9'/>"
    "<circle cx='20' cy='15' r='7' fill='white' opacity='0.9'/>"
    "<ellipse cx='20' cy='34' rx='11' ry='8' fill='white' opacity='0.9'/>"
    "</svg>"
)
BOT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'>"
    "<circle cx='20' cy='20' r='20' fill='%231e3a5f'/>"
    "<rect x='10' y='13' width='20' height='14' rx='4' fill='white' opacity='0.9'/>"
    "<circle cx='15' cy='20' r='2.5' fill='%231e3a5f'/>"
    "<circle cx='25' cy='20' r='2.5' fill='%231e3a5f'/>"
    "<rect x='17' y='27' width='6' height='3' rx='1' fill='white' opacity='0.9'/>"
    "<rect x='18' y='8' width='4' height='6' rx='2' fill='white' opacity='0.9'/>"
    "<circle cx='20' cy='8' r='2' fill='%234A90D9'/>"
    "</svg>"
)

# =============================================================================
#  LOAD ARTIFACTS
# =============================================================================

print("=" * 60)
print("  Loading AI Copilot components...")
print("=" * 60)

with open(f'{ARTIFACTS_DIR}/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open(f'{ARTIFACTS_DIR}/best_model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open(f'{ARTIFACTS_DIR}/knowledge_base.pkl', 'rb') as f:
    KNOWLEDGE_BASE = pickle.load(f)
with open(f'{ARTIFACTS_DIR}/chunk_index.pkl', 'rb') as f:
    chunk_index = pickle.load(f)

print(f"  [OK] Classifier : {type(clf).__name__}")
print(f"  [OK] Knowledge base: {len(KNOWLEDGE_BASE)} intents")

embedder    = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
faiss_index = faiss.read_index(f'{ARTIFACTS_DIR}/faiss_index.bin')
print(f"  [OK] Embedder + FAISS ({faiss_index.ntotal} vectors)")

chroma_client = chromadb.PersistentClient(path=f'{ARTIFACTS_DIR}/chroma_db')
collection    = chroma_client.get_collection("clinc150_kb")
print(f"  [OK] Chroma: {collection.count()} docs")
print("=" * 60)

# =============================================================================
#  GUARDRAILS
# =============================================================================

BLOCKED_PATTERNS = [
    r'\b(hack|exploit|bypass|jailbreak|ignore\s+previous|ignore\s+above|disregard\s+instructions|forget\s+instructions)\b',
    r'\b(bomb|weapon|kill|attack|terror|suicide)\b',
    r'<script|javascript:|onclick=|onerror=',
    r'(select\s+\*\s+from|drop\s+table|insert\s+into)',
]

OFF_TOPIC_PHRASES = [
    'election', 'vote for', 'sports score', 'movie review',
    'song lyrics', 'homework help', 'write my essay',
    'medical advice', 'diagnosis', 'prescription',
    'stock tips', 'crypto advice',
]

CLARIFICATION_PHRASES = [
    "didn't understand", "don't understand", "not clear", "confused",
    "can you explain", "what do you mean", "please clarify", "elaborate",
    "i'm lost", "can you repeat", "didn't get that", "please explain again",
    "rephrase", "simpler", "in simple terms", "i don't get it",
    "not helpful", "still confused", "couldn't understand",
]

def guardrail_input(text: str) -> Tuple[bool, str]:
    if len(text.strip()) < MIN_INPUT_LEN:
        return False, "Please enter a valid query (at least 3 characters)."
    if len(text) > MAX_INPUT_LEN:
        return False, f"Query too long ({len(text)} chars). Please keep it under {MAX_INPUT_LEN} characters."
    t = text.lower()
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return False, ("🛡️ This request contains content that violates usage policy. "
                           "Please ask a genuine banking or service-related question.")
    for phrase in OFF_TOPIC_PHRASES:
        if phrase in t:
            return False, (f"🛡️ I'm a customer support assistant for banking and financial services. "
                           f"I can't help with '{phrase}'. "
                           f"Please ask about your account, cards, travel, or other banking services.")
    return True, ""

def guardrail_output(response: str, retrieved_docs: List[Dict]) -> Tuple[str, List[str]]:
    warnings = []
    pii_patterns = [
        r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
        r'\b\d{9,18}\b',
    ]
    for pat in pii_patterns:
        if re.search(pat, response):
            warnings.append("⚠️ Response may contain sensitive numbers — review before sending.")
            break
    if retrieved_docs:
        all_text   = " ".join([d.get('resolution','') for d in retrieved_docs]).lower()
        r_words    = set(re.findall(r'\b[a-z]{4,}\b', response.lower()))
        kb_words   = set(re.findall(r'\b[a-z]{4,}\b', all_text))
        overlap    = len(r_words & kb_words) / max(len(r_words), 1)
        if overlap < 0.12:
            warnings.append("⚠️ Low faithfulness — verify response is grounded in KB.")
    if len(response.split()) > 160:
        response = " ".join(response.split()[:160]) + "..."
        warnings.append("ℹ️ Response trimmed to 160 words.")
    return response, warnings

def is_clarification(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in CLARIFICATION_PHRASES)

def detect_drift(current_domain: str, state: List[Dict]) -> Tuple[bool, str]:
    recent = [s.get('domain','') for s in state[-DRIFT_WINDOW:]
              if s.get('role') == 'assistant' and s.get('domain')]
    if not recent:
        return False, ""
    most_common = max(set(recent), key=recent.count)
    if current_domain and current_domain != most_common and most_common not in ('', 'General'):
        return True, f"{most_common} → {current_domain}"
    return False, ""

# =============================================================================
#  CONFIDENCE RESCALING
#  WHY: LinearSVC.decision_function returns raw margin scores. After softmax
#  over 150 classes, the max probability is tiny (~0.02-0.15) because the
#  softmax denominator sums 150 terms. This makes ALL queries look low
#  confidence and the filter always falls to "none".
#
#  FIX: Rank-based rescaling. The argmax class's rank among 150 scores
#  is always rank 1 (by definition). We use the margin gap between the
#  top-1 and top-2 scores, normalized by the score range, to get a
#  meaningful [0,1] confidence. Large gap = high confidence. Small gap =
#  ambiguous. This correctly reflects classifier certainty.
# =============================================================================

def predict_intent(query: str) -> Tuple[str, float, Dict]:
    vec  = tfidf.transform([_clean(query)])
    pred = clf.predict(vec)[0]

    if hasattr(clf, 'decision_function'):
        raw    = clf.decision_function(vec)[0]          # shape (150,)
        sorted_scores = np.sort(raw)[::-1]
        top1   = sorted_scores[0]
        top2   = sorted_scores[1]
        gap    = top1 - top2                             # margin gap
        span   = raw.max() - raw.min()
        # Normalize gap by score span → [0, 1]
        conf   = float(np.clip(gap / max(span, 1e-6), 0, 1))

        classes  = clf.classes_
        top5_idx = np.argsort(raw)[::-1][:5]
        # Show top-5 as relative scores for display
        top5_raw = raw[top5_idx]
        top5_rel = (top5_raw - top5_raw.min()) / max(top5_raw.max() - top5_raw.min(), 1e-6)
        top5     = {classes[i]: float(top5_rel[j]) for j, i in enumerate(top5_idx)}

    elif hasattr(clf, 'predict_proba'):
        proba    = clf.predict_proba(vec)[0]
        conf     = float(np.max(proba))
        classes  = clf.classes_
        top5_idx = np.argsort(proba)[::-1][:5]
        top5     = {classes[i]: float(proba[i]) for i in top5_idx}
    else:
        conf = 0.8
        top5 = {pred: 0.8}

    return pred, conf, top5

def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return re.sub(r"\s+", ' ', text).strip()

# =============================================================================
#  RETRIEVAL
# =============================================================================

def retrieve_docs(query: str, intent: str, conf: float) -> Tuple[List[Dict], str]:
    q_emb  = embedder.encode([query], normalize_embeddings=True).tolist()
    domain = KNOWLEDGE_BASE.get(intent, {}).get("domain", "General")

    if conf >= CONF_HIGH and intent in KNOWLEDGE_BASE:
        where_filter = {"intent": {"$eq": intent}}
        mode         = "intent"
    elif conf >= CONF_LOW:
        where_filter = {"domain": {"$eq": domain}}
        mode         = "domain"
    else:
        where_filter = None
        mode         = "none"

    try:
        if where_filter:
            result = collection.query(
                query_embeddings=q_emb,
                n_results=min(TOP_K, collection.count()),
                where=where_filter
            )
        else:
            result = collection.query(query_embeddings=q_emb, n_results=TOP_K)
    except Exception:
        result = collection.query(query_embeddings=q_emb, n_results=TOP_K)
        mode   = "fallback"

    docs = []
    for did, dtxt, meta, dist in zip(
        result['ids'][0], result['documents'][0],
        result['metadatas'][0], result['distances'][0]
    ):
        docs.append({
            "intent":   meta["intent"],
            "domain":   meta["domain"],
            "title":    meta["title"],
            "resolution": KNOWLEDGE_BASE.get(meta["intent"], {}).get("resolution", dtxt),
            "policy":     KNOWLEDGE_BASE.get(meta["intent"], {}).get("policy", ""),
            "score":    float(1 - dist),
            "mode":     mode,
        })
    return docs, mode

# =============================================================================
#  LLM GENERATION  (Groq or Gemini, selectable in UI)
# =============================================================================

def _build_prompt(query: str, docs: List[Dict], state: List[Dict],
                  drift: bool, drift_msg: str, clarify: bool) -> Tuple[str, List[Dict]]:
    ctx = "\n\n".join([
        f"[Doc {i+1}: {d['title']} — {d['domain']}]\n{d['resolution']}\nPolicy: {d['policy']}"
        for i, d in enumerate(docs)
    ])
    drift_note   = f"\nNOTE: Topic shifted ({drift_msg}). Address the new topic.\n" if drift else ""
    clarify_note = ("\nIMPORTANT: Customer did not understand previous reply. "
                    "Rephrase simply with a concrete example.\n") if clarify else ""

    system = (
        "You are an AI customer support copilot for a bank and financial services company. "
        "Generate accurate, concise (3–5 sentences), empathetic suggested replies for support agents.\n\n"
        "RULES:\n"
        "1. Use ONLY information from the retrieved documents below.\n"
        "2. Never guess or hallucinate policies or numbers.\n"
        "3. If documents lack relevant info, say: 'I don't have specific info — connect customer to a specialist.'\n"
        "4. Fraud/security: always direct to 24/7 helpline immediately.\n"
        "5. Plain text only — no markdown, bullets, or asterisks.\n"
        "6. Start directly with resolution. No filler phrases.\n"
        "7. Address customer as 'you'/'your'. End with offer to help further.\n"
        f"{drift_note}{clarify_note}\n"
        f"KNOWLEDGE BASE:\n{ctx}"
    )
    # Build conversation history for context memory
    messages = [{"role": "system", "content": system}]
    recent = [s for s in state if s.get('role') in ('user','assistant')][-MAX_HISTORY:]
    for s in recent:
        messages.append({"role": s['role'], "content": s['content']})
    messages.append({"role": "user", "content": query})
    return system, messages

def generate(query: str, docs: List[Dict], state: List[Dict],
             drift: bool, drift_msg: str, clarify: bool,
             model_label: str, groq_key: str, gemini_key: str) -> Dict:

    system_str, messages = _build_prompt(query, docs, state, drift, drift_msg, clarify)

    # Determine provider
    is_gemini = model_label in GEMINI_MODELS
    model_id  = GEMINI_MODELS.get(model_label) or GROQ_MODELS.get(model_label, "llama-3.1-8b-instant")

    if is_gemini:
        try:
            import google.generativeai as genai
            key = gemini_key.strip() or os.environ.get("GOOGLE_API_KEY","")
            if not key:
                return {"response":"No Gemini API key provided.","success":False,"model":model_id,"tokens":0}
            genai.configure(api_key=key)
            m   = genai.GenerativeModel(model_id)
            # Gemini doesn't support system role in messages — inject into user turn
            prompt = system_str + "\n\nCustomer query: " + query
            resp   = m.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=300, temperature=0.3)
            )
            return {"response": resp.text.strip(), "success": True, "model": model_id, "tokens": 0}
        except Exception as e:
            return {"response": f"Gemini error: {e}", "success": False, "model": model_id, "tokens": 0}
    else:
        try:
            key = groq_key.strip() or os.environ.get("GROQ_API_KEY","")
            if not key:
                return {"response":"No Groq API key provided.","success":False,"model":model_id,"tokens":0}
            client   = Groq(api_key=key)
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=300,
                temperature=0.3,
                top_p=0.9,
            )
            text   = response.choices[0].message.content.strip()
            tokens = response.usage.completion_tokens
            return {"response": text, "success": True, "model": model_id, "tokens": tokens}
        except Exception as e:
            return {"response": f"Groq error: {e}", "success": False, "model": model_id, "tokens": 0}

def quality_metrics(generated: str, docs: List[Dict]) -> Dict:
    rouge  = rouge_scorer.RougeScorer(['rougeL','rouge1'], use_stemmer=True)
    ref    = docs[0]['resolution'] if docs else ""
    sc     = rouge.score(ref, generated)
    rw     = set(re.findall(r'\b[a-z]{4,}\b', ref.lower()))
    gw     = set(re.findall(r'\b[a-z]{4,}\b', generated.lower()))
    faith  = len(rw & gw) / max(len(rw), 1)
    return {
        "rougeL":      round(sc['rougeL'].fmeasure, 3),
        "rouge1":      round(sc['rouge1'].fmeasure, 3),
        "faithfulness": round(faith, 3),
        "words":       len(generated.split()),
    }

# =============================================================================
#  PANEL HTML BUILDERS
# =============================================================================

def _intent_html(intent: str, domain: str, conf: float, mode: str, top5: Dict) -> str:
    pct   = int(conf * 100)
    color = "#27ae60" if pct >= 55 else "#f39c12" if pct >= 25 else "#e74c3c"
    mode_label = {
        "intent":   ("Intent filter active", "#0c447c", "#e6f1fb"),
        "domain":   ("Domain filter active", "#085041", "#e1f5ee"),
        "none":     ("No filter — open search", "#633806", "#faeeda"),
        "fallback": ("Filter fallback", "#7B2D8B", "#f3e8f9"),
    }.get(mode, ("Unknown", "#555", "#f5f5f5"))

    rows = ""
    for nm, val in list(top5.items())[:5]:
        w = int(val * 150)
        rows += (f"<tr><td style='padding:2px 8px;font-size:12px;font-family:monospace'>{nm}</td>"
                 f"<td><div style='background:#e9ecef;border-radius:3px;height:10px;width:150px'>"
                 f"<div style='background:#2d6a9f;width:{w}px;height:10px;border-radius:3px'></div>"
                 f"</div></td>"
                 f"<td style='padding:0 8px;font-size:11px;color:#666'>{val:.3f}</td></tr>")

    return f"""
<div style='font-family:sans-serif;padding:10px'>
  <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>
    <span style='background:#e6f1fb;border:1px solid #b5d4f4;border-radius:16px;
                 padding:4px 14px;font-size:13px;font-weight:600;color:#0c447c'>🎯 {intent}</span>
    <span style='background:#e1f5ee;border:1px solid #9fe1cb;border-radius:16px;
                 padding:4px 14px;font-size:13px;font-weight:600;color:#085041'>🏷 {domain}</span>
    <span style='background:{mode_label[2]};border:1px solid {mode_label[1]};border-radius:16px;
                 padding:4px 12px;font-size:11px;font-weight:600;color:{mode_label[1]}'>🔍 {mode_label[0]}</span>
  </div>
  <div style='margin-bottom:8px'>
    <span style='font-size:12px;color:#555'>Confidence: <b style='color:{color}'>{pct}%</b>
    &nbsp;{"→ intent filter" if pct>=55 else "→ domain filter" if pct>=25 else "→ no filter"}</span>
    <div style='background:#e9ecef;border-radius:5px;height:7px;width:100%;max-width:280px;margin-top:4px'>
      <div style='background:{color};width:{min(pct,100)}%;height:7px;border-radius:5px'></div>
    </div>
  </div>
  <details style='margin-top:6px'>
    <summary style='font-size:12px;color:#2d6a9f;cursor:pointer'>▸ Top-5 intent predictions</summary>
    <table style='margin-top:6px;border-collapse:collapse'>{rows}</table>
    <p style='font-size:11px;color:#999;margin:4px 0 0'>
      Scores are relative margin values (LinearSVC). Higher = more confident.
    </p>
  </details>
</div>"""

def _docs_html(docs: List[Dict], mode: str) -> str:
    if not docs:
        return "<p style='color:#aaa;padding:10px'>No documents retrieved.</p>"
    mode_desc = {
        "intent":   "filtered to predicted intent",
        "domain":   "filtered to predicted domain",
        "none":     "full knowledge base search",
        "fallback": "filter failed — full search",
    }.get(mode, mode)
    cards = ""
    for i, d in enumerate(docs, 1):
        sc  = int(d['score'] * 100)
        col = "#27ae60" if sc >= 70 else "#f39c12" if sc >= 40 else "#e74c3c"
        res = d['resolution'][:220] + ('…' if len(d['resolution']) > 220 else '')
        pol = f"<p style='font-size:11px;color:#999;margin:5px 0 0'><b>Policy:</b> {d['policy'][:110]}</p>" if d.get('policy') else ""
        cards += f"""
<div style='border:1px solid #e2e8f0;border-left:4px solid #2d6a9f;border-radius:7px;
            padding:10px 12px;margin:6px 0;background:#f8fafc'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>
    <b style='color:#1e3a5f;font-size:13px'>#{i} {d["title"]}</b>
    <span style='font-size:11px;background:#edf2f7;border-radius:8px;padding:2px 7px;color:{col}'>
      {d["score"]:.3f}
    </span>
  </div>
  <span style='background:#e1f5ee;color:#085041;border-radius:8px;padding:2px 8px;font-size:11px'>{d["domain"]}</span>
  <p style='margin:6px 0 0;font-size:12px;color:#4a5568;line-height:1.5'>{res}</p>
  {pol}
</div>"""
    return f"<div style='padding:6px'><p style='font-size:11px;color:#999;margin:0 0 4px'>Retrieval: <b>{mode_desc}</b> | {len(docs)} docs</p>{cards}</div>"

def _metrics_html(q: Dict, t: Dict, llm: Dict) -> str:
    def col(v, lo=0.2, hi=0.4):
        return "#27ae60" if v >= hi else "#f39c12" if v >= lo else "#e74c3c"
    rows = [
        ("ROUGE-L",      q['rougeL'],       col(q['rougeL'])),
        ("ROUGE-1",      q['rouge1'],       col(q['rouge1'])),
        ("Faithfulness", q['faithfulness'], col(q['faithfulness'])),
        ("Word count",   q['words'],        "#555"),
        ("Classify ms",  t['c'],            "#555"),
        ("Retrieve ms",  t['r'],            "#555"),
        ("LLM ms",       t['l'],            "#555"),
        ("Total ms",     t['tot'],          "#27ae60"),
    ]
    trs = ""
    for i,(label,val,color) in enumerate(rows):
        bg = "background:#f7fafc;" if i%2==0 else ""
        trs += (f"<tr style='{bg}'>"
                f"<td style='padding:6px 8px;font-size:12px'>{label}</td>"
                f"<td style='padding:6px 8px;text-align:right;font-weight:600;color:{color}'>{val}</td>"
                f"</tr>")
    return (f"<div style='padding:8px'>"
            f"<table style='width:100%;border-collapse:collapse'>{trs}"
            f"<tr><td style='padding:6px 8px;font-size:11px;color:#999'>Model</td>"
            f"<td style='padding:6px 8px;text-align:right;font-size:11px;color:#999'>{llm.get('model','')}</td></tr>"
            f"</table></div>")

def _backend_log(query, intent, domain, conf, mode, docs, llm, q, t,
                 drift, drift_msg, warns, clarify) -> str:
    ts    = datetime.now().strftime("%H:%M:%S")
    lines = [
        f"{'='*54}", f"  PIPELINE TRACE  [{ts}]", f"{'='*54}", "",
        f"[INPUT GUARDRAILS]   PASSED", "",
        f"[STEP 1 — ML CLASSIFICATION]",
        f"  Query        : {query[:80]}",
        f"  Intent       : {intent}",
        f"  Domain       : {domain}",
        f"  Confidence   : {conf:.4f}  ({int(conf*100)}%)",
        f"  Filter route : {'intent filter (conf>=0.55)' if conf>=CONF_HIGH else 'domain filter (conf>=0.25)' if conf>=CONF_LOW else 'no filter (conf<0.25)'}",
        f"  Classify ms  : {t['c']}",
        "", f"[STEP 2 — RETRIEVAL]",
        f"  Filter mode  : {mode}",
        f"  Docs found   : {len(docs)}",
    ]
    for i,d in enumerate(docs,1):
        lines.append(f"  Doc {i}: {d['title']} ({d['domain']}) score={d['score']:.3f}")
    lines += [
        f"  Retrieve ms  : {t['r']}", "",
        f"[STEP 3 — LLM GENERATION]",
        f"  Model        : {llm.get('model','')}",
        f"  Output tokens: {llm.get('tokens',0)}",
        f"  LLM ms       : {t['l']}",
        f"  Success      : {llm.get('success',False)}", "",
        f"[STEP 4 — OUTPUT GUARDRAILS]",
        f"  ROUGE-L      : {q['rougeL']}",
        f"  Faithfulness : {q['faithfulness']}",
        f"  Words        : {q['words']}",
    ]
    if warns:
        lines.append(f"  WARNINGS     : {' | '.join(warns)}")
    lines += [
        "", f"[CONTEXT & DRIFT]",
        f"  Topic drift  : {'YES — ' + drift_msg if drift else 'No'}",
        f"  Clarify mode : {'YES' if clarify else 'No'}",
        "", f"  TOTAL MS     : {t['tot']}",
        f"{'='*54}",
    ]
    return "\n".join(lines)

# =============================================================================
#  MAIN CHAT FUNCTION
# =============================================================================

def chat(user_msg: str, history: List, state: List,
         model_label: str, groq_key: str, gemini_key: str):

    t0 = time.time()

    # ── INPUT GUARDRAILS ──────────────────────────────────────────────────
    safe, reject_msg = guardrail_input(user_msg)
    if not safe:
        bot = reject_msg
        history.append({"role":"user","content":user_msg})
        history.append({"role":"assistant","content":bot})
        state.append({"role":"assistant","content":bot,"domain":"BLOCKED","intent":"BLOCKED"})
        log = f"[GUARDRAIL BLOCKED]\n  Reason: {reject_msg}"
        ih  = _intent_html("BLOCKED","N/A",0.0,"none",{})
        dh  = "<p style='color:#e74c3c;padding:10px'>Query blocked by input guardrails.</p>"
        mh  = "<p style='color:#aaa;padding:10px'>N/A</p>"
        return "", history, state, log, ih, dh, mh

    clarify = is_clarification(user_msg)

    # ── STEP 1: CLASSIFY ─────────────────────────────────────────────────
    t1      = time.time()
    intent, conf, top5 = predict_intent(user_msg)
    domain  = KNOWLEDGE_BASE.get(intent, {}).get("domain", "General")
    c_ms    = int((time.time()-t1)*1000)

    drift, drift_msg = detect_drift(domain, state)

    # ── STEP 2: RETRIEVE ─────────────────────────────────────────────────
    t2      = time.time()
    docs, mode = retrieve_docs(user_msg, intent, conf)
    r_ms    = int((time.time()-t2)*1000)

    # ── STEP 3: GENERATE ─────────────────────────────────────────────────
    t3      = time.time()
    llm     = generate(user_msg, docs, state, drift, drift_msg, clarify,
                       model_label, groq_key, gemini_key)
    l_ms    = int((time.time()-t3)*1000)
    tot_ms  = int((time.time()-t0)*1000)

    response = llm["response"]

    # ── OUTPUT GUARDRAILS ─────────────────────────────────────────────────
    response, warns = guardrail_output(response, docs)

    # ── QUALITY ───────────────────────────────────────────────────────────
    q = quality_metrics(response, docs)
    t = {"c": c_ms, "r": r_ms, "l": l_ms, "tot": tot_ms}

    # Build display response
    display = response
    if warns:
        display += "\n\n" + "\n".join(warns)
    if drift:
        display = f"[Topic shift: {drift_msg}]\n\n{display}"

    # Update history and state
    history.append({"role":"user",      "content":user_msg})
    history.append({"role":"assistant", "content":display})
    state.append({"role":"user",      "content":user_msg,  "domain":domain,"intent":intent})
    state.append({"role":"assistant", "content":display,   "domain":domain,"intent":intent,
                  "confidence":conf, "mode":mode, "quality":q})
    if len(state) > MAX_HISTORY*2+4:
        state = state[-(MAX_HISTORY*2):]

    # Build panels
    log = _backend_log(user_msg,intent,domain,conf,mode,docs,llm,q,t,
                       drift,drift_msg,warns,clarify)
    ih  = _intent_html(intent, domain, conf, mode, top5)
    dh  = _docs_html(docs, mode)
    mh  = _metrics_html(q, t, llm)

    print(log)
    return "", history, state, log, ih, dh, mh

def clear_all(history, state):
    return [], [], "Cleared.", "<p style='color:#aaa;padding:10px'>Cleared.</p>", \
           "<p style='color:#aaa;padding:10px'>Cleared.</p>", \
           "<p style='color:#aaa;padding:10px'>Cleared.</p>"

# =============================================================================
#  GRADIO UI
# =============================================================================

SAMPLE_QUERIES = [
    "My payment failed but money was deducted",
    "I want to increase my credit card limit",
    "How do I book a flight to Mumbai?",
    "Someone made an unauthorised transaction on my card",
    "I couldn't understand your previous answer, explain simply",
    "What is my credit score and how to improve it?",
    "Tell me who will win the next election",     # OUT OF SCOPE — guardrail demo
    "How do I check my account balance?",
]

# ── Theme: dark/light handled by Gradio's built-in theme toggle ──────────
LIGHT_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)
DARK_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

# We use a single Blocks build; Gradio 4.x supports js-based dark mode toggle
with gr.Blocks(
    theme=LIGHT_THEME,
    title="AI Support Copilot",
    css="""
    .dark-mode { filter: invert(0.92) hue-rotate(180deg); }
    .dark-mode img { filter: invert(1) hue-rotate(180deg); }
    footer { display:none !important; }
    .guardrail-note {
        font-size:11px; color:#718096; margin-top:6px; padding:8px;
        background:#f7fafc; border-radius:6px; border:1px solid #e2e8f0;
    }
    """
) as demo:

    state_var    = gr.State([])
    dark_mode_js = """
    () => {
        const el = document.querySelector('.gradio-container');
        el.classList.toggle('dark-mode');
    }
    """

    # ── HEADER ────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML("""
            <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2d6a9f 100%);
                        padding:18px 24px;border-radius:10px;color:white">
              <h1 style="margin:0;font-size:22px;font-weight:600">
                🤖 AI Customer Support Copilot
              </h1>
              <p style="margin:5px 0 0;font-size:13px;opacity:0.85">
                Week 3 · CLINC150 · LinearSVC + Chroma RAG + Groq/Gemini LLM · Guardrails
              </p>
            </div>""")
        with gr.Column(scale=1, min_width=100):
            theme_btn = gr.Button("🌙 Dark", size="sm", variant="secondary")
            theme_btn.click(fn=None, js=dark_mode_js)

    # ── SETTINGS ROW ─────────────────────────────────────────────────────
    with gr.Row():
        model_dd = gr.Dropdown(
            choices=ALL_MODEL_LABELS,
            value=DEFAULT_MODEL_LABEL,
            label="LLM Model",
            scale=3,
        )
        groq_key_box = gr.Textbox(
            label="Groq API Key",
            placeholder="gsk_... (console.groq.com — free)",
            type="password",
            scale=2,
            value=os.environ.get("GROQ_API_KEY",""),
        )
        gemini_key_box = gr.Textbox(
            label="Gemini API Key (optional)",
            placeholder="AIza... (aistudio.google.com — free)",
            type="password",
            scale=2,
            value=os.environ.get("GOOGLE_API_KEY",""),
        )

    # ── MAIN ROW: Chat + Debug ────────────────────────────────────────────
    with gr.Row():

        # LEFT — Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=500,
                bubble_full_width=False,
                type="messages",
                avatar_images=(USER_AVATAR, BOT_AVATAR),
                show_label=False,
            )
            with gr.Row():
                msg_box  = gr.Textbox(
                    placeholder="Type your customer support query here...",
                    label="",
                    lines=2,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

            gr.HTML("<p style='font-size:12px;color:#718096;margin:6px 0 3px'>Quick test queries:</p>")
            with gr.Row():
                sb1 = gr.Button(SAMPLE_QUERIES[0][:38], size="sm")
                sb2 = gr.Button(SAMPLE_QUERIES[1][:38], size="sm")
                sb3 = gr.Button(SAMPLE_QUERIES[2][:38], size="sm")
                sb4 = gr.Button(SAMPLE_QUERIES[3][:38], size="sm")
            with gr.Row():
                sb5 = gr.Button(SAMPLE_QUERIES[4][:38], size="sm")
                sb6 = gr.Button(SAMPLE_QUERIES[5][:38], size="sm")
                sb7 = gr.Button("🚫 Out-of-scope demo", size="sm", variant="stop")
                sb8 = gr.Button(SAMPLE_QUERIES[7][:38], size="sm")

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear Chat", size="sm", variant="secondary")

            gr.HTML("""<div class='guardrail-note'>
              <b>Active guardrails:</b> Input validation · Prompt injection · Off-topic blocking ·
              PII output scan · Faithfulness check · Topic drift detection · Clarification mode
            </div>""")

        # RIGHT — Debug panels
        with gr.Column(scale=2):
            gr.HTML("<div style='font-size:13px;font-weight:600;color:#1e3a5f;margin-bottom:6px'>Pipeline Debug</div>")
            with gr.Accordion("🎯 Intent & Classification", open=True):
                intent_panel = gr.HTML(
                    "<p style='color:#aaa;font-size:13px;padding:8px'>Send a query to see results.</p>")
            with gr.Accordion("📄 Retrieved Documents", open=True):
                docs_panel = gr.HTML(
                    "<p style='color:#aaa;font-size:13px;padding:8px'>Docs appear here.</p>")
            with gr.Accordion("📊 Quality Metrics & Latency", open=False):
                metrics_panel = gr.HTML(
                    "<p style='color:#aaa;font-size:13px;padding:8px'>Metrics after first query.</p>")

    # ── BACKEND LOG ───────────────────────────────────────────────────────
    with gr.Accordion("🖥 Backend Pipeline Log", open=False):
        backend_log = gr.Code(
            value="Backend log appears here after first query...",
            language="python",
            lines=26,
            label="",
        )

    # ── EVENT WIRING ─────────────────────────────────────────────────────
    OUTPUTS = [msg_box, chatbot, state_var, backend_log,
               intent_panel, docs_panel, metrics_panel]
    INPUTS  = [msg_box, chatbot, state_var, model_dd, groq_key_box, gemini_key_box]

    send_btn.click(fn=chat, inputs=INPUTS, outputs=OUTPUTS)
    msg_box.submit(fn=chat, inputs=INPUTS, outputs=OUTPUTS)

    # Sample buttons — direct text injection
    sb1.click(fn=lambda: SAMPLE_QUERIES[0], outputs=msg_box)
    sb2.click(fn=lambda: SAMPLE_QUERIES[1], outputs=msg_box)
    sb3.click(fn=lambda: SAMPLE_QUERIES[2], outputs=msg_box)
    sb4.click(fn=lambda: SAMPLE_QUERIES[3], outputs=msg_box)
    sb5.click(fn=lambda: SAMPLE_QUERIES[4], outputs=msg_box)
    sb6.click(fn=lambda: SAMPLE_QUERIES[5], outputs=msg_box)
    sb7.click(fn=lambda: SAMPLE_QUERIES[6], outputs=msg_box)   # out-of-scope
    sb8.click(fn=lambda: SAMPLE_QUERIES[7], outputs=msg_box)

    clear_btn.click(
        fn=clear_all,
        inputs=[chatbot, state_var],
        outputs=[chatbot, state_var, backend_log,
                 intent_panel, docs_panel, metrics_panel],
    )

# =============================================================================
#  LAUNCH
# =============================================================================
if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        show_error=True,
        server_port=7860,
        inbrowser=False,
    )
