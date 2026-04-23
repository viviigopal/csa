# =============================================================================
# WEEK 3 — AI CUSTOMER SUPPORT COPILOT (GRADIO + GROQ) - STABLE & FIXED
# Fixed ChromaDB '_id' error + llama-3.1-8b-instant
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import re
import pickle
import time
import numpy as np
from typing import List, Dict
import gradio as gr

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from rouge_score import rouge_scorer

ARTIFACTS_PATH = "/content/drive/MyDrive/saved_models/artifacts"

# ========================== LOAD ARTIFACTS ==========================
def load_artifacts():
    print("Loading artifacts...")
    with open(f'{ARTIFACTS_PATH}/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/knowledge_base.pkl', 'rb') as f:
        knowledge_base = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/chunk_index.pkl', 'rb') as f:
        chunk_index = pickle.load(f)

    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = faiss.read_index(f'{ARTIFACTS_PATH}/faiss_index.bin')
    
    chroma_client = chromadb.PersistentClient(path=f'{ARTIFACTS_PATH}/chroma_db')
    collection = chroma_client.get_collection("clinc150_kb")

    print("✅ All artifacts loaded successfully!")
    return tfidf, clf, embedder, faiss_index, collection, knowledge_base, chunk_index

tfidf, clf, embedder, faiss_index, collection, knowledge_base, chunk_index = load_artifacts()

# ========================== CORE FUNCTIONS ==========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return re.sub(r"\s+", ' ', text).strip()

def predict_intent(query: str):
    vec = tfidf.transform([clean_text(query)])
    pred = clf.predict(vec)[0]
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(vec)[0]
        conf = float(np.max(proba))
    else:
        conf = 0.85
    return str(pred), conf

def retrieve_documents(query: str, predicted_intent: str, confidence: float, top_k: int = 3, use_filter: bool = True):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    predicted_domain = knowledge_base.get(predicted_intent, {}).get("domain", "General")

    if use_filter:
        if confidence >= 0.70:
            where = {"intent": {"$eq": predicted_intent}}
            filter_mode = "intent"
        elif confidence >= 0.40:
            where = {"domain": {"$eq": predicted_domain}}
            filter_mode = "domain"
        else:
            where = None
            filter_mode = "none"
    else:
        where = None
        filter_mode = "no_filter"

    try:
        result = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            where=where
        ) if where else collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k
        )
    except Exception:
        result = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
        filter_mode = "fallback"

    docs = []
    # FIXED: Safe access to ChromaDB results
    ids = result.get('ids', [[]])[0]
    documents = result.get('documents', [[]])[0]
    metadatas = result.get('metadatas', [[]])[0]
    distances = result.get('distances', [[]])[0]

    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc_text = documents[i] if i < len(documents) else ""
        
        docs.append({
            "title": meta.get("title", "Unknown"),
            "domain": meta.get("domain", "General"),
            "resolution": knowledge_base.get(meta.get("intent"), {}).get("resolution", doc_text),
            "retrieval_score": float(1 - distances[i]) if i < len(distances) else 0.0,
            "filter_mode": filter_mode
        })
    return docs, filter_mode

def generate_with_groq(query: str, retrieved_docs: List[Dict], groq_key: str):
    if not groq_key or groq_key.strip() == "":
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Please contact customer support."
        return fallback, False

    system_prompt = """You are a professional bank customer support copilot.
Give concise, empathetic, and accurate replies (3-5 sentences).
Use ONLY the information from the documents below."""

    context = "\n\n".join([f"[{d['title']} — {d['domain']}]\n{d['resolution']}" for d in retrieved_docs])

    try:
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"KNOWLEDGE BASE:\n{context}\n\nQUERY: {query}\n\nSuggested Reply:"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip(), True
    except Exception as e:
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Our team will help you."
        return f"⚠️ Groq Error: {str(e)[:80]}...\n\nFallback: {fallback}", False

# ========================== GRADIO APP ==========================
def process_query(query, groq_key, top_k, use_filter):
    if not query.strip():
        return "Please enter a query", None, None, None, None, []

    start = time.time()
    pred_intent, confidence = predict_intent(query)
    retrieved_docs, filter_mode = retrieve_documents(query, pred_intent, confidence, int(top_k), use_filter)
    llm_response, success = generate_with_groq(query, retrieved_docs, groq_key)
    total_time = round((time.time() - start) * 1000)

    result = {
        "query": query,
        "predicted_intent": pred_intent,
        "confidence": confidence,
        "filter_mode": filter_mode,
        "retrieved_docs": retrieved_docs,
        "llm_response": llm_response,
        "timings": total_time,
        "success": success
    }

    status = f"✅ Done in {total_time}ms | Intent: **{pred_intent}** ({confidence:.1%})"
    return status, result, retrieved_docs, llm_response, [], []

def approve(current_result, edited_response, history):
    if not current_result:
        return history, "Nothing to approve"
    entry = {**current_result, "final_response": edited_response, "status": "approved", "timestamp": time.strftime("%H:%M:%S")}
    history.append(entry)
    return history, "✅ Approved!"

def reject(current_result, history):
    if not current_result:
        return history, "Nothing to reject"
    entry = {**current_result, "final_response": "", "status": "rejected", "timestamp": time.strftime("%H:%M:%S")}
    history.append(entry)
    return history, "❌ Rejected"

with gr.Blocks(title="AI Support Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Customer Support Copilot\n**Stable Version • llama-3.1-8b-instant**")

    with gr.Tab("💬 Copilot"):
        groq_key = gr.Textbox(label="Groq API Key", type="password", placeholder="gsk_...")

        with gr.Row():
            query_input = gr.Textbox(label="Customer Query", lines=3, placeholder="My payment failed but money was deducted...")
            with gr.Column():
                top_k = gr.Slider(1, 5, value=3, label="Top-K Documents")
                use_filter = gr.Checkbox(True, label="Use Intent Filter")

        run_btn = gr.Button("🚀 Run Pipeline", variant="primary")

        status = gr.Markdown("Ready")
        current_result = gr.State(None)
        history_state = gr.State([])

        with gr.Row():
            docs_out = gr.JSON(label="Retrieved Documents")
            response_box = gr.Textbox(label="AI Suggested Reply (editable)", lines=8)

        with gr.Row():
            approve_btn = gr.Button("✅ Approve & Send", variant="primary")
            reject_btn = gr.Button("❌ Reject")

        # Sample Queries
        with gr.Row():
            for q in ["Payment failed but money deducted", "My account is blocked", "Unauthorized transaction", "How to check balance?"]:
                gr.Button(q).click(lambda x=q: x, outputs=query_input)

        run_btn.click(
            process_query,
            inputs=[query_input, groq_key, top_k, use_filter],
            outputs=[status, current_result, docs_out, response_box, [], history_state]
        )

        approve_btn.click(approve, inputs=[current_result, response_box, history_state], outputs=[history_state, status])
        reject_btn.click(reject, inputs=[current_result, history_state], outputs=[history_state, status])

    with gr.Tab("📖 History"):
        gr.Markdown("✅ Approve or Reject to see history here")

    gr.Markdown("---\nGet free Groq key at: https://console.groq.com/keys")

demo.launch(share=True)
