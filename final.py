# =============================================================================
#  WEEK 3 — AI CUSTOMER SUPPORT COPILOT  (Gradio + Groq)
#  Works in Google Colab without tunneling
#
#  INSTALL (one cell):
#  pip install gradio groq sentence-transformers faiss-cpu chromadb
#              rouge-score scikit-learn xgboost imbalanced-learn datasets
#
#  GROQ FREE API KEY:  https://console.groq.com  (free, 14400 req/day)
#  MODEL USED: llama-3.1-8b-instant  (fastest free model, low latency)
#
#  RUN:
#  python week3_app.py
#  OR in Colab:  !python week3_app.py
#
#  The app auto-opens in Colab's output cell.
#  share=True gives a public URL for demo (no tunneling needed).
# =============================================================================

import os, re, json, pickle, time, textwrap
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import gradio as gr
from groq import Groq

from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from rouge_score import rouge_scorer

# =============================================================================
#  CONFIGURATION
# =============================================================================

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
ARTIFACTS_DIR   = "artifacts"

# Groq model — llama-3.1-8b-instant chosen because:
# - 14,400 free requests/day (vs Gemini 15 RPM)
# - Sub-second latency (~0.3s)
# - Strong instruction following
# - No rate-limit issues for demos
GROQ_MODEL      = "llama-3.1-8b-instant"

# Pipeline thresholds
CONF_HIGH       = 0.70   # above → intent-level filter
CONF_LOW        = 0.40   # above → domain-level filter, below → no filter
TOP_K           = 3      # retrieved documents

# Guardrail settings
MAX_INPUT_LEN   = 500    # max characters per user message
MIN_INPUT_LEN   = 3      # min characters to process
MAX_HISTORY     = 10     # conversation turns to keep in memory

# Topic drift: if predicted domain differs from last N turns' domain, flag it
DRIFT_WINDOW    = 3      # turns to look back for drift detection

# =============================================================================
#  LOAD ALL ARTIFACTS (once at startup)
# =============================================================================

print("=" * 60)
print("  Loading AI Copilot components...")
print("=" * 60)

def load_artifacts():
    """Load Week 1 + Week 2 artifacts."""
    with open(f'{ARTIFACTS_DIR}/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/intent_list.pkl', 'rb') as f:
        intent_list = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/knowledge_base.pkl', 'rb') as f:
        knowledge_base = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/chunk_index.pkl', 'rb') as f:
        chunk_index = pickle.load(f)
    return tfidf, clf, le, intent_list, knowledge_base, chunk_index

tfidf, clf, le, intent_list, KNOWLEDGE_BASE, chunk_index = load_artifacts()
print(f"  [OK] ML artifacts: {type(clf).__name__}, {len(intent_list)} intents")

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  [OK] Embedder: all-MiniLM-L6-v2")

faiss_index = faiss.read_index(f'{ARTIFACTS_DIR}/faiss_index.bin')
print(f"  [OK] FAISS index: {faiss_index.ntotal} vectors")

chroma_client = chromadb.PersistentClient(path=f'{ARTIFACTS_DIR}/chroma_db')
collection = chroma_client.get_collection("clinc150_kb")
print(f"  [OK] Chroma: {collection.count()} documents")

groq_client = Groq(api_key=GROQ_API_KEY)
print(f"  [OK] Groq client: model={GROQ_MODEL}")
print("=" * 60)

# =============================================================================
#  GUARDRAILS
# =============================================================================

# Input blocklist — queries that should be refused
BLOCKED_PATTERNS = [
    r'\b(hack|exploit|bypass|jailbreak|ignore previous|ignore above|disregard|forget instructions)\b',
    r'\b(bomb|weapon|kill|attack|terror)\b',
    r'<script|javascript:|onclick=|onerror=',    # XSS attempts
    r'(select\s+\*\s+from|drop\s+table|insert\s+into)',  # SQL injection
]

# Off-topic domains: queries the copilot should redirect
OFF_TOPIC_PHRASES = [
    'politics', 'election', 'vote', 'sports score', 'movie',
    'song lyrics', 'recipe for', 'homework', 'essay', 'poem about',
    'covid', 'medical advice', 'diagnosis', 'prescription',
]

# Banking/support topic keywords — used for topic drift detection
BANKING_KEYWORDS = {
    'Banking', 'Credit Cards', 'Travel', 'Utilities', 'Home',
    'Auto & Commute', 'Small Talk', 'Meta', 'Shopping', 'General'
}

def guardrail_input(text: str) -> Tuple[bool, str]:
    """
    INPUT GUARDRAILS:
    1. Length check
    2. Blocked patterns (prompt injection, harmful content)
    3. Off-topic detection
    Returns (is_safe, rejection_message_or_empty)
    """
    if len(text.strip()) < MIN_INPUT_LEN:
        return False, "Please enter a valid query (at least 3 characters)."

    if len(text) > MAX_INPUT_LEN:
        return False, f"Query too long ({len(text)} chars). Please keep it under {MAX_INPUT_LEN} characters."

    text_lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False, ("⚠️ I cannot process that request. This appears to contain "
                          "content that violates usage policy. Please ask a genuine support question.")

    for phrase in OFF_TOPIC_PHRASES:
        if phrase in text_lower:
            return False, (f"I'm a customer support assistant for banking and financial services. "
                          f"I'm not able to help with '{phrase}' topics. "
                          f"Please ask about your account, transactions, cards, travel bookings, or other services.")

    return True, ""

def guardrail_output(response: str, retrieved_docs: List[Dict]) -> Tuple[str, List[str]]:
    """
    OUTPUT GUARDRAILS:
    1. Hallucination check — flag if response contains information not in retrieved docs
    2. PII leakage check — block if response contains card/account number patterns
    3. Response length check
    Returns (validated_response, list_of_warnings)
    """
    warnings = []

    # PII leakage — block card/account numbers
    pii_patterns = [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # card number
        r'\b\d{9,18}\b',                                    # account number
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',             # SSN-like
    ]
    for pat in pii_patterns:
        if re.search(pat, response):
            warnings.append("⚠️ Response may contain sensitive numbers — please review before sending.")
            break

    # Hallucination proxy — check if key factual claims in response
    # appear to be grounded in retrieved docs
    if retrieved_docs:
        all_retrieved_text = " ".join([d.get('resolution', '') + " " + d.get('policy', '')
                                       for d in retrieved_docs]).lower()
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response.lower()))
        retrieved_words = set(re.findall(r'\b[a-z]{4,}\b', all_retrieved_text))
        overlap = len(response_words & retrieved_words) / max(len(response_words), 1)
        if overlap < 0.15:
            warnings.append("⚠️ Low grounding score — response may contain information not in knowledge base. Verify before sending.")

    # Length check
    if len(response.split()) < 5:
        warnings.append("⚠️ Response is very short. Consider elaborating.")
    if len(response.split()) > 150:
        response = " ".join(response.split()[:150]) + "..."
        warnings.append("ℹ️ Response trimmed to 150 words for conciseness.")

    return response, warnings

def detect_topic_drift(current_domain: str, history: List[Dict]) -> Tuple[bool, str]:
    """
    TOPIC DRIFT DETECTION:
    Look at last DRIFT_WINDOW turns. If current domain is different
    from all recent turns' domains, flag it as a topic change.
    This helps the LLM understand context shifts.
    """
    if len(history) < 2:
        return False, ""

    recent_domains = [h.get('domain', '') for h in history[-DRIFT_WINDOW:]
                      if h.get('role') == 'assistant' and h.get('domain')]

    if not recent_domains:
        return False, ""

    # If current domain different from majority of recent domains
    most_common = max(set(recent_domains), key=recent_domains.count)
    if current_domain != most_common and most_common != '' and current_domain != 'General':
        return True, f"Topic changed from {most_common} to {current_domain}"

    return False, ""

# =============================================================================
#  CORE PIPELINE FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return re.sub(r"\s+", ' ', text).strip()

def predict_intent(query: str) -> Tuple[str, float, Dict]:
    """
    WEEK 1 CLASSIFIER
    Returns intent, confidence, and top-5 predictions dict.
    """
    vec = tfidf.transform([clean_text(query)])
    pred = clf.predict(vec)[0]

    if hasattr(clf, 'decision_function'):
        scores = clf.decision_function(vec)[0]
        exp_s  = np.exp(scores - np.max(scores))
        proba  = exp_s / exp_s.sum()
        conf   = float(np.max(proba))
        classes = clf.classes_
        top5_idx = np.argsort(proba)[::-1][:5]
        top5 = {classes[i]: float(proba[i]) for i in top5_idx}
    elif hasattr(clf, 'predict_proba'):
        proba  = clf.predict_proba(vec)[0]
        conf   = float(np.max(proba))
        classes = clf.classes_
        top5_idx = np.argsort(proba)[::-1][:5]
        top5 = {classes[i]: float(proba[i]) for i in top5_idx}
    else:
        conf = 0.8
        top5 = {pred: 0.8}

    return pred, conf, top5

def retrieve_documents(query: str, predicted_intent: str, confidence: float,
                       top_k: int = TOP_K) -> Tuple[List[Dict], str]:
    """
    WEEK 2 RETRIEVAL — Chroma with confidence-based intent filtering.
    confidence >= 0.70 → intent filter (tight)
    confidence >= 0.40 → domain filter (medium)
    confidence <  0.40 → no filter (open search)
    """
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    predicted_domain = KNOWLEDGE_BASE.get(predicted_intent, {}).get("domain", "General")

    if confidence >= CONF_HIGH and predicted_intent in KNOWLEDGE_BASE:
        where_filter  = {"intent": {"$eq": predicted_intent}}
        filter_mode   = "intent"
    elif confidence >= CONF_LOW:
        where_filter  = {"domain": {"$eq": predicted_domain}}
        filter_mode   = "domain"
    else:
        where_filter  = None
        filter_mode   = "none"

    try:
        if where_filter:
            result = collection.query(
                query_embeddings=q_emb,
                n_results=min(top_k, collection.count()),
                where=where_filter
            )
        else:
            result = collection.query(query_embeddings=q_emb, n_results=top_k)
    except Exception:
        result = collection.query(query_embeddings=q_emb, n_results=top_k)
        filter_mode = "fallback"

    docs = []
    for doc_id, doc_text, meta, dist in zip(
        result['ids'][0], result['documents'][0],
        result['metadatas'][0], result['distances'][0]
    ):
        docs.append({
            "chunk_id":       doc_id,
            "intent":         meta["intent"],
            "domain":         meta["domain"],
            "title":          meta["title"],
            "text":           doc_text,
            "resolution":     KNOWLEDGE_BASE.get(meta["intent"], {}).get("resolution", doc_text),
            "policy":         KNOWLEDGE_BASE.get(meta["intent"], {}).get("policy", ""),
            "retrieval_score": float(1 - dist),
            "filter_mode":    filter_mode,
        })
    return docs, filter_mode

def build_groq_messages(query: str, retrieved_docs: List[Dict],
                        history: List[Dict], drift_flag: bool,
                        drift_msg: str, clarification_mode: bool) -> List[Dict]:
    """
    Build message list for Groq chat API.
    Includes:
    - System prompt with guardrails
    - Conversation history (last MAX_HISTORY turns) for context memory
    - Topic drift notice if detected
    - Clarification instruction if user said they didn't understand
    - Retrieved documents as context
    """
    # Build context from retrieved docs
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[Doc {i}: {doc['title']} — {doc['domain']}]\n"
            f"Resolution: {doc['resolution']}\n"
            f"Policy: {doc['policy']}"
        )
    context_str = "\n\n".join(context_parts)

    drift_note = ""
    if drift_flag:
        drift_note = f"\nNOTE: The customer's topic has changed ({drift_msg}). Address the new topic directly.\n"

    clarification_note = ""
    if clarification_mode:
        clarification_note = (
            "\nIMPORTANT: The customer said they did not understand the previous answer. "
            "Rephrase your last response in simpler language with a concrete example if possible. "
            "Do not repeat the same wording.\n"
        )

    system_prompt = f"""You are an AI customer support copilot for a bank and financial services company.
Your task is to generate accurate, concise (3–5 sentences), and empathetic suggested replies for support agents.

STRICT RULES:
1. Use ONLY information from the retrieved knowledge base documents below.
2. Never guess, hallucinate, or make up policies, numbers, or procedures.
3. If the documents do not contain relevant information, say: "I don't have specific information on that — please connect the customer with a specialist."
4. For fraud/security issues: always direct to 24/7 fraud helpline immediately.
5. Keep response to 3–5 sentences maximum. Professional and empathetic tone.
6. Plain text only — no markdown, no bullet points, no asterisks.
7. Start directly with the resolution. No filler phrases.
8. Address the customer as "you" and "your".
9. End with an offer for further assistance.
{drift_note}{clarification_note}
RETRIEVED KNOWLEDGE BASE:
{context_str}
"""
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history for context memory (last MAX_HISTORY turns)
    recent = [h for h in history if h.get('role') in ('user', 'assistant')][-MAX_HISTORY:]
    for h in recent:
        if h['role'] in ('user', 'assistant'):
            messages.append({"role": h['role'], "content": h['content']})

    # Current query
    messages.append({"role": "user", "content": query})
    return messages

def generate_with_groq(messages: List[Dict]) -> Dict:
    """
    Call Groq API with llama-3.1-8b-instant.
    WHY GROQ over Gemini:
    - 14,400 free requests/day vs Gemini's 15 RPM (≈900/day)
    - ~0.3s latency vs ~1.5s for Gemini
    - No rate-limit errors during demo
    - llama-3.1-8b-instant is specifically optimised for low-latency inference
    """
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,    # Low temp = factual, consistent
            top_p=0.9,
        )
        text = response.choices[0].message.content.strip()
        usage = response.usage
        return {
            "response": text,
            "success": True,
            "model": GROQ_MODEL,
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
    except Exception as e:
        return {
            "response": f"LLM unavailable: {e}",
            "success": False,
            "model": GROQ_MODEL,
            "input_tokens": 0,
            "output_tokens": 0,
        }

def compute_quality_metrics(generated: str, retrieved_docs: List[Dict]) -> Dict:
    """Compute ROUGE-L and faithfulness proxy for the generated response."""
    rouge = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
    reference = retrieved_docs[0]['resolution'] if retrieved_docs else ""
    sc = rouge.score(reference, generated)

    ref_words = set(re.findall(r'\b[a-z]{4,}\b', reference.lower()))
    gen_words = set(re.findall(r'\b[a-z]{4,}\b', generated.lower()))
    faithfulness = len(ref_words & gen_words) / max(len(ref_words), 1)

    return {
        "rougeL":      round(sc['rougeL'].fmeasure, 4),
        "rouge1":      round(sc['rouge1'].fmeasure, 4),
        "faithfulness": round(faithfulness, 4),
        "word_count":  len(generated.split()),
    }

# =============================================================================
#  CLARIFICATION DETECTOR
# =============================================================================

CLARIFICATION_PHRASES = [
    "didn't understand", "don't understand", "not clear", "confused",
    "can you explain", "what do you mean", "please clarify", "elaborate",
    "i'm lost", "can you repeat", "didn't get that", "please explain again",
    "rephrase", "simpler", "in simple terms", "what does that mean",
    "i don't get it", "not helpful", "still confused"
]

def is_clarification_request(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in CLARIFICATION_PHRASES)

# =============================================================================
#  BACKEND LOG (printed to terminal + shown in UI)
# =============================================================================

def format_backend_log(query, intent, domain, conf, filter_mode,
                       docs, llm_result, quality, timings,
                       drift_flag, drift_msg, warnings,
                       clarification_mode) -> str:
    """Format a detailed backend trace for display in UI debug panel."""
    ts = datetime.now().strftime("%H:%M:%S")
    log_lines = [
        f"{'='*56}",
        f"  PIPELINE TRACE  [{ts}]",
        f"{'='*56}",
        f"",
        f"[INPUT GUARDRAILS]   PASSED",
        f"",
        f"[STEP 1 — ML CLASSIFICATION]",
        f"  Query         : {query[:80]}",
        f"  Predicted intent: {intent}",
        f"  Domain        : {domain}",
        f"  Confidence    : {conf:.4f}  ({conf*100:.1f}%)",
        f"  Routing       : {'intent filter' if conf>=CONF_HIGH else 'domain filter' if conf>=CONF_LOW else 'no filter'}",
        f"  Classify time : {timings['classify_ms']}ms",
        f"",
        f"[STEP 2 — RETRIEVAL]",
        f"  Filter mode   : {filter_mode}",
        f"  Documents     : {len(docs)} retrieved",
    ]
    for i, d in enumerate(docs, 1):
        log_lines.append(f"  Doc {i}: {d['title']} ({d['domain']}) — score: {d['retrieval_score']:.4f}")
    log_lines += [
        f"  Retrieve time : {timings['retrieve_ms']}ms",
        f"",
        f"[STEP 3 — LLM GENERATION]",
        f"  Model         : {llm_result.get('model', GROQ_MODEL)}",
        f"  Input tokens  : {llm_result.get('input_tokens', 'N/A')}",
        f"  Output tokens : {llm_result.get('output_tokens', 'N/A')}",
        f"  LLM time      : {timings['llm_ms']}ms",
        f"  LLM success   : {llm_result.get('success', False)}",
        f"",
        f"[STEP 4 — OUTPUT GUARDRAILS]",
        f"  ROUGE-L       : {quality['rougeL']}",
        f"  Faithfulness  : {quality['faithfulness']}",
        f"  Word count    : {quality['word_count']}",
    ]
    if warnings:
        log_lines.append(f"  WARNINGS      : {' | '.join(warnings)}")
    log_lines += [
        f"",
        f"[CONTEXT & DRIFT]",
        f"  Topic drift   : {'YES — ' + drift_msg if drift_flag else 'No'}",
        f"  Clarification : {'YES — rephrasing' if clarification_mode else 'No'}",
        f"",
        f"[TOTAL PIPELINE TIME]   {timings['total_ms']}ms",
        f"{'='*56}",
    ]
    return "\n".join(log_lines)

# =============================================================================
#  MAIN CHAT FUNCTION (called by Gradio on every message)
# =============================================================================

def chat(user_message: str, history: List[List], backend_state: List[Dict],
         api_key_state: str) -> Tuple:
    """
    Main pipeline function.
    Parameters:
      user_message  : current user input
      history       : Gradio chat history [[user, bot], ...]
      backend_state : list of per-turn pipeline metadata dicts
      api_key_state : Groq API key from textbox

    Returns:
      ("", updated_history, updated_backend_state,
       backend_log_text, intent_badge_html, docs_html,
       metrics_html, eval_chart_fig)
    """
    if api_key_state.strip():
        groq_client.api_key = api_key_state.strip()
        os.environ["GROQ_API_KEY"] = api_key_state.strip()

    t_start = time.time()

    # ── INPUT GUARDRAILS ───────────────────────────────────────────────────
    is_safe, rejection_msg = guardrail_input(user_message)
    if not is_safe:
        bot_msg = f"🛡️ {rejection_msg}"
        history.append([user_message, bot_msg])
        backend_state.append({
            "role": "assistant", "content": bot_msg,
            "domain": "N/A", "intent": "BLOCKED",
        })
        log = f"[GUARDRAIL BLOCKED]\n  Reason: {rejection_msg}"
        intent_html = _intent_html("BLOCKED", "N/A", 0.0, "intent", {})
        docs_html   = "<p style='color:#888'>Query blocked by guardrails.</p>"
        metrics_html= "<p style='color:#888'>N/A</p>"
        return "", history, backend_state, log, intent_html, docs_html, metrics_html, None

    clarification_mode = is_clarification_request(user_message)
    t_classify_start   = time.time()

    # ── STEP 1: ML CLASSIFICATION ──────────────────────────────────────────
    predicted_intent, confidence, top5 = predict_intent(user_message)
    predicted_domain = KNOWLEDGE_BASE.get(predicted_intent, {}).get("domain", "General")
    t_classify_ms = int((time.time() - t_classify_start) * 1000)

    # ── TOPIC DRIFT DETECTION ──────────────────────────────────────────────
    drift_flag, drift_msg = detect_topic_drift(predicted_domain, backend_state)

    # ── STEP 2: RETRIEVAL ─────────────────────────────────────────────────
    t_ret_start = time.time()
    retrieved_docs, filter_mode = retrieve_documents(
        user_message, predicted_intent, confidence, TOP_K
    )
    t_retrieve_ms = int((time.time() - t_ret_start) * 1000)

    # ── STEP 3: LLM GENERATION ────────────────────────────────────────────
    t_llm_start = time.time()
    messages    = build_groq_messages(
        user_message, retrieved_docs, backend_state,
        drift_flag, drift_msg, clarification_mode
    )
    llm_result  = generate_with_groq(messages)
    t_llm_ms    = int((time.time() - t_llm_start) * 1000)

    raw_response = llm_result["response"]

    # ── STEP 4: OUTPUT GUARDRAILS ─────────────────────────────────────────
    validated_response, out_warnings = guardrail_output(raw_response, retrieved_docs)

    # ── QUALITY METRICS ───────────────────────────────────────────────────
    quality = compute_quality_metrics(validated_response, retrieved_docs)

    t_total_ms = int((time.time() - t_start) * 1000)
    timings    = {
        "classify_ms": t_classify_ms,
        "retrieve_ms": t_retrieve_ms,
        "llm_ms":      t_llm_ms,
        "total_ms":    t_total_ms,
    }

    # ── FORMAT CHATBOT RESPONSE ───────────────────────────────────────────
    # Show warnings inline if any
    display_response = validated_response
    if out_warnings:
        display_response += "\n\n" + "\n".join(out_warnings)
    if drift_flag:
        display_response = f"[Topic shift detected: {drift_msg}]\n\n{display_response}"

    # Update history + state
    history.append([user_message, display_response])
    backend_state.append({
        "role":    "user",
        "content": user_message,
        "domain":  predicted_domain,
        "intent":  predicted_intent,
    })
    backend_state.append({
        "role":          "assistant",
        "content":       display_response,
        "domain":        predicted_domain,
        "intent":        predicted_intent,
        "confidence":    confidence,
        "filter_mode":   filter_mode,
        "quality":       quality,
    })

    # Trim state to avoid unbounded growth
    if len(backend_state) > MAX_HISTORY * 2:
        backend_state = backend_state[-(MAX_HISTORY * 2):]

    # ── BUILD UI PANELS ───────────────────────────────────────────────────
    backend_log  = format_backend_log(
        user_message, predicted_intent, predicted_domain, confidence,
        filter_mode, retrieved_docs, llm_result, quality, timings,
        drift_flag, drift_msg, out_warnings, clarification_mode
    )
    intent_html  = _intent_html(predicted_intent, predicted_domain, confidence, filter_mode, top5)
    docs_html    = _docs_html(retrieved_docs, filter_mode)
    metrics_html = _metrics_html(quality, timings, llm_result)
    eval_fig     = _build_eval_chart(backend_state)

    print(backend_log)   # also print to terminal / Colab cell

    return "", history, backend_state, backend_log, intent_html, docs_html, metrics_html, eval_fig

# =============================================================================
#  UI PANEL BUILDERS
# =============================================================================

def _intent_html(intent: str, domain: str, conf: float, filter_mode: str, top5: Dict) -> str:
    conf_pct   = int(conf * 100)
    conf_color = "#27ae60" if conf_pct >= 70 else "#f39c12" if conf_pct >= 40 else "#e74c3c"
    filter_colors = {
        "intent": ("#0c447c", "#e6f1fb"),
        "domain": ("#085041", "#e1f5ee"),
        "none":   ("#633806", "#faeeda"),
        "fallback": ("#7B2D8B", "#f3e8f9"),
    }
    fc, fb = filter_colors.get(filter_mode, ("#555", "#f5f5f5"))

    top5_rows = ""
    for int_name, prob in list(top5.items())[:5]:
        bar_w = int(prob * 160)
        top5_rows += (
            f"<tr><td style='padding:3px 8px;font-size:12px;font-family:monospace'>{int_name}</td>"
            f"<td><div style='background:#e9ecef;border-radius:4px;height:12px;width:160px'>"
            f"<div style='background:#2d6a9f;width:{bar_w}px;height:12px;border-radius:4px'></div>"
            f"</div></td>"
            f"<td style='padding:0 8px;font-size:12px;color:#555'>{prob:.4f}</td></tr>"
        )

    html = f"""
<div style="font-family:sans-serif;padding:12px">
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">
    <div style="background:#e6f1fb;border:1px solid #b5d4f4;border-radius:20px;
                padding:5px 16px;font-size:13px;font-weight:600;color:#0c447c">
      🎯 {intent}
    </div>
    <div style="background:#e1f5ee;border:1px solid #9fe1cb;border-radius:20px;
                padding:5px 16px;font-size:13px;font-weight:600;color:#085041">
      🏷️ {domain}
    </div>
    <div style="background:{fb};border:1px solid {fc};border-radius:20px;
                padding:5px 16px;font-size:12px;font-weight:600;color:{fc}">
      🔍 Filter: {filter_mode}
    </div>
  </div>
  <div style="margin-bottom:6px">
    <div style="font-size:12px;color:#555;margin-bottom:3px">
      Confidence: <b style="color:{conf_color}">{conf_pct}%</b>
      {"&nbsp;— HIGH (intent filter)" if conf_pct>=70 else "&nbsp;— MEDIUM (domain filter)" if conf_pct>=40 else "&nbsp;— LOW (no filter)"}
    </div>
    <div style="background:#e9ecef;border-radius:6px;height:8px;width:100%;max-width:300px">
      <div style="background:{conf_color};width:{conf_pct}%;height:8px;border-radius:6px"></div>
    </div>
  </div>
  <details style="margin-top:8px">
    <summary style="font-size:12px;color:#2d6a9f;cursor:pointer">Top-5 predictions</summary>
    <table style="margin-top:6px;border-collapse:collapse">{top5_rows}</table>
  </details>
</div>"""
    return html

def _docs_html(docs: List[Dict], filter_mode: str) -> str:
    if not docs:
        return "<p style='color:#888;padding:12px'>No documents retrieved.</p>"

    cards = ""
    for i, d in enumerate(docs, 1):
        score_pct = int(d['retrieval_score'] * 100)
        score_color = "#27ae60" if score_pct >= 70 else "#f39c12" if score_pct >= 40 else "#e74c3c"
        cards += f"""
<div style="border:1px solid #e2e8f0;border-left:4px solid #2d6a9f;border-radius:8px;
            padding:12px 14px;margin:8px 0;background:#f8fafc;font-family:sans-serif">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <b style="color:#1e3a5f;font-size:14px">#{i} {d['title']}</b>
    <span style="font-size:11px;background:#edf2f7;border-radius:10px;padding:2px 8px;color:{score_color}">
      score: {d['retrieval_score']:.3f}
    </span>
  </div>
  <span style="background:#e1f5ee;color:#085041;border-radius:10px;
               padding:2px 8px;font-size:11px">{d['domain']}</span>
  <p style="margin:8px 0 0;font-size:13px;color:#4a5568;line-height:1.5">
    {d['resolution'][:250]}{'...' if len(d['resolution'])>250 else ''}
  </p>
  {'<p style="margin:6px 0 0;font-size:11px;color:#718096"><b>Policy:</b> ' + d["policy"][:120] + '</p>' if d.get("policy") else ''}
</div>"""

    return f"""
<div style="padding:8px">
  <p style="font-size:12px;color:#718096;margin:0 0 6px">
    Retrieved {len(docs)} doc(s) | Filter: <b>{filter_mode}</b>
  </p>
  {cards}
</div>"""

def _metrics_html(quality: Dict, timings: Dict, llm_result: Dict) -> str:
    def color_val(v, low=0.2, high=0.4):
        return "#27ae60" if v>=high else "#f39c12" if v>=low else "#e74c3c"

    return f"""
<div style="font-family:sans-serif;padding:10px">
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f7fafc">
      <th style="padding:8px;text-align:left;font-size:12px;color:#718096;border-bottom:1px solid #e2e8f0">Metric</th>
      <th style="padding:8px;text-align:right;font-size:12px;color:#718096;border-bottom:1px solid #e2e8f0">Value</th>
    </tr>
    <tr>
      <td style="padding:7px 8px;font-size:13px">ROUGE-L</td>
      <td style="padding:7px 8px;text-align:right;font-weight:600;color:{color_val(quality['rougeL'])}">{quality['rougeL']}</td>
    </tr>
    <tr style="background:#f7fafc">
      <td style="padding:7px 8px;font-size:13px">ROUGE-1</td>
      <td style="padding:7px 8px;text-align:right;font-weight:600;color:{color_val(quality['rouge1'])}">{quality['rouge1']}</td>
    </tr>
    <tr>
      <td style="padding:7px 8px;font-size:13px">Faithfulness</td>
      <td style="padding:7px 8px;text-align:right;font-weight:600;color:{color_val(quality['faithfulness'])}">{quality['faithfulness']}</td>
    </tr>
    <tr style="background:#f7fafc">
      <td style="padding:7px 8px;font-size:13px">Word count</td>
      <td style="padding:7px 8px;text-align:right;color:#555">{quality['word_count']}</td>
    </tr>
    <tr>
      <td style="padding:7px 8px;font-size:13px">Classify (ms)</td>
      <td style="padding:7px 8px;text-align:right;color:#555">{timings['classify_ms']}</td>
    </tr>
    <tr style="background:#f7fafc">
      <td style="padding:7px 8px;font-size:13px">Retrieve (ms)</td>
      <td style="padding:7px 8px;text-align:right;color:#555">{timings['retrieve_ms']}</td>
    </tr>
    <tr>
      <td style="padding:7px 8px;font-size:13px">LLM (ms)</td>
      <td style="padding:7px 8px;text-align:right;color:#555">{timings['llm_ms']}</td>
    </tr>
    <tr style="background:#edf7f0">
      <td style="padding:7px 8px;font-size:13px;font-weight:600">Total (ms)</td>
      <td style="padding:7px 8px;text-align:right;font-weight:700;color:#27ae60">{timings['total_ms']}</td>
    </tr>
    <tr>
      <td style="padding:7px 8px;font-size:12px;color:#718096">Model</td>
      <td style="padding:7px 8px;text-align:right;font-size:12px;color:#718096">{llm_result.get('model','groq')}</td>
    </tr>
  </table>
</div>"""

def _build_eval_chart(backend_state: List[Dict]) -> Optional[plt.Figure]:
    """Build a live evaluation chart from conversation history."""
    turns = [s for s in backend_state if s.get('role') == 'assistant' and 'quality' in s]
    if len(turns) < 2:
        return None

    turn_nums    = list(range(1, len(turns) + 1))
    rouge_vals   = [t['quality']['rougeL']       for t in turns]
    faith_vals   = [t['quality']['faithfulness']  for t in turns]
    conf_vals    = [t.get('confidence', 0)        for t in turns]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.patch.set_facecolor('#f8fafc')

    for ax, vals, label, color in zip(
        axes,
        [rouge_vals, faith_vals, conf_vals],
        ['ROUGE-L', 'Faithfulness', 'Confidence'],
        ['#4C72B0', '#55A868', '#DD8452']
    ):
        ax.plot(turn_nums, vals, 'o-', color=color, linewidth=2, markersize=6)
        ax.fill_between(turn_nums, vals, alpha=0.15, color=color)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_xlabel('Turn', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(turn_nums)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_facecolor('#f8fafc')
        sns.despine(ax=ax)

    plt.tight_layout()
    return fig

def clear_chat(history, backend_state):
    return [], [], "", "", "", "", None

def use_sample(sample_text, history, backend_state):
    """Inject a sample query into the input box."""
    return sample_text

# =============================================================================
#  GRADIO UI
# =============================================================================

SAMPLE_QUERIES = [
    "My payment failed but money was deducted from my account",
    "I want to increase my credit card limit",
    "My account is blocked and I can't log in",
    "How do I book a flight to Mumbai?",
    "Someone made an unauthorised transaction on my card",
    "What is my credit score and how to improve it?",
    "I couldn't understand your previous answer, please explain simply",
    "How do I check my account balance?",
]

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(theme=THEME, title="AI Support Copilot") as demo:

    # ── State ─────────────────────────────────────────────────────────────
    backend_state = gr.State([])   # per-turn metadata
    api_key_state = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2d6a9f 100%);
                padding:20px 28px;border-radius:12px;margin-bottom:8px;color:white">
      <h1 style="margin:0;font-size:26px;font-weight:600">
        🤖 AI Customer Support Copilot
      </h1>
      <p style="margin:6px 0 0;font-size:14px;opacity:0.85">
        Week 3 — ML Classification + RAG + Groq LLaMA | CLINC150 | With Guardrails & Context Memory
      </p>
    </div>
    """)

    with gr.Row():
        # ── LEFT COLUMN: Chat ──────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Customer Support Conversation",
                height=480,
                bubble_full_width=False,
                show_label=True,
                avatar_images=("👤", "🤖"),
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your customer support query here...",
                    label="",
                    lines=2,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

            # Sample queries
            gr.HTML("<p style='font-size:12px;color:#718096;margin:6px 0 4px'>Quick test queries:</p>")
            with gr.Row():
                sample_btns = [
                    gr.Button(label=q[:35]+"…" if len(q)>35 else q, size="sm")
                    for q in SAMPLE_QUERIES[:4]
                ]
            with gr.Row():
                sample_btns2 = [
                    gr.Button(label=q[:35]+"…" if len(q)>35 else q, size="sm")
                    for q in SAMPLE_QUERIES[4:]
                ]

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear Chat", size="sm", variant="secondary")

            gr.HTML("""
            <div style="font-size:11px;color:#718096;margin-top:6px;padding:8px;
                        background:#f7fafc;border-radius:6px;border:1px solid #e2e8f0">
              <b>Guardrails active:</b> Input validation · Prompt injection detection ·
              Off-topic blocking · Output PII scan · Faithfulness check · Topic drift detection · Clarification mode
            </div>""")

        # ── RIGHT COLUMN: Debug Panels ─────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML("<b style='font-size:14px;color:#1e3a5f'>Pipeline Debug</b>")

            with gr.Accordion("🎯 Intent & Classification", open=True):
                intent_panel = gr.HTML("<p style='color:#aaa;font-size:13px;padding:10px'>Send a query to see classification results.</p>")

            with gr.Accordion("📄 Retrieved Documents", open=True):
                docs_panel = gr.HTML("<p style='color:#aaa;font-size:13px;padding:10px'>Retrieved docs will appear here.</p>")

            with gr.Accordion("📊 Quality Metrics & Latency", open=False):
                metrics_panel = gr.HTML("<p style='color:#aaa;font-size:13px;padding:10px'>Metrics appear after first query.</p>")

    # ── API Key Row ────────────────────────────────────────────────────────
    with gr.Row():
        api_key_box = gr.Textbox(
            label="Groq API Key (paste here or set GROQ_API_KEY env var)",
            placeholder="gsk_...",
            type="password",
            scale=3,
        )
        gr.HTML("""
        <div style="padding:8px 0;font-size:12px;color:#555">
          Get free key: <a href="https://console.groq.com" target="_blank">console.groq.com</a><br>
          Free tier: 14,400 req/day · ~0.3s latency
        </div>""")

    # ── Backend Log ────────────────────────────────────────────────────────
    with gr.Accordion("🖥 Backend Pipeline Log (full trace)", open=False):
        backend_log = gr.Code(
            label="",
            language="python",
            lines=28,
            value="Backend log will appear here after first query..."
        )

    # ── Evaluation Charts ──────────────────────────────────────────────────
    with gr.Accordion("📈 Live Evaluation Charts (ROUGE-L · Faithfulness · Confidence)", open=False):
        eval_chart = gr.Plot(label="")

    # ── Week 2 Eval Summary ────────────────────────────────────────────────
    with gr.Accordion("📋 Week 2 Retrieval Evaluation Summary", open=False):
        try:
            ret_summary = pd.read_csv(f"{ARTIFACTS_DIR}/retrieval_summary.csv")
            gen_summary = pd.read_csv(f"{ARTIFACTS_DIR}/generation_eval_results.csv")
            abl_summary = pd.read_csv(f"{ARTIFACTS_DIR}/ablation_results.csv")
            eval_html = f"""
<div style="font-family:sans-serif;padding:10px">
  <h4 style="color:#1e3a5f;margin-top:0">Retrieval Evaluation (FAISS vs Chroma)</h4>
  {ret_summary.to_html(index=False, border=0, classes='eval-table')}
  <h4 style="color:#1e3a5f;margin-top:16px">Generation Quality (n=50 samples)</h4>
  <table style="border-collapse:collapse">
    <tr><td style="padding:4px 12px">ROUGE-L</td><td style="padding:4px 12px;font-weight:600;color:#2d6a9f">{gen_summary['rougeL'].mean():.4f}</td></tr>
    <tr><td style="padding:4px 12px">Faithfulness</td><td style="padding:4px 12px;font-weight:600;color:#2d6a9f">{gen_summary['faithfulness'].mean():.4f}</td></tr>
  </table>
  <h4 style="color:#1e3a5f;margin-top:16px">Ablation: Intent Filter Contribution</h4>
  <table style="border-collapse:collapse">
    <tr><td style="padding:4px 12px">With filter</td><td style="padding:4px 12px;font-weight:600;color:#27ae60">{abl_summary['with_filter'].mean():.4f}</td></tr>
    <tr><td style="padding:4px 12px">Without filter</td><td style="padding:4px 12px;font-weight:600;color:#e74c3c">{abl_summary['without_filter'].mean():.4f}</td></tr>
  </table>
</div>"""
        except Exception as e:
            eval_html = f"<p style='color:#888'>Run week2_rag_gemini.py first to generate eval CSVs. ({e})</p>"
        gr.HTML(eval_html)

    # ── EVENT WIRING ───────────────────────────────────────────────────────
    OUTPUTS = [msg_box, chatbot, backend_state, backend_log,
               intent_panel, docs_panel, metrics_panel, eval_chart]

    def send_wrapper(msg, history, bs, ak):
        return chat(msg, history, bs, ak)

    send_btn.click(
        fn=send_wrapper,
        inputs=[msg_box, chatbot, backend_state, api_key_box],
        outputs=OUTPUTS,
    )
    msg_box.submit(
        fn=send_wrapper,
        inputs=[msg_box, chatbot, backend_state, api_key_box],
        outputs=OUTPUTS,
    )

    # Sample query buttons
    for btn, q in zip(sample_btns, SAMPLE_QUERIES[:4]):
        btn.click(fn=lambda query=q: query, outputs=msg_box)
    for btn, q in zip(sample_btns2, SAMPLE_QUERIES[4:]):
        btn.click(fn=lambda query=q: query, outputs=msg_box)

    clear_btn.click(
        fn=clear_chat,
        inputs=[chatbot, backend_state],
        outputs=[chatbot, backend_state, backend_log,
                 intent_panel, docs_panel, metrics_panel, eval_chart]
    )

    api_key_box.change(fn=lambda k: k, inputs=api_key_box, outputs=api_key_state)

# =============================================================================
#  LAUNCH
# =============================================================================
if __name__ == "__main__":
    demo.launch(
        share=True,           # gives public URL — works in Colab without tunneling
        debug=True,
        show_error=True,
        server_port=7860,
        inbrowser=False,      # False for Colab (no local browser)
    )
