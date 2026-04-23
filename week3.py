# =============================================================================
# WEEK 3 — AI CUSTOMER SUPPORT COPILOT (GRADIO + GROQ)
# Fixed Version - No Streamlit references
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import re
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple

import gradio as gr

# Groq
from groq import Groq

# RAG
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from rouge_score import rouge_scorer

# ========================== PATH ==========================
ARTIFACTS_PATH = "/content/drive/MyDrive/saved_models/artifacts"

# ========================== LOAD ARTIFACTS ==========================
def load_artifacts():
    print("Loading artifacts...")
    with open(f'{ARTIFACTS_PATH}/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/intent_list.pkl', 'rb') as f:
        intent_list = pickle.load(f)

    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = faiss.read_index(f'{ARTIFACTS_PATH}/faiss_index.bin')

    chroma_client = chromadb.PersistentClient(path=f'{ARTIFACTS_PATH}/chroma_db')
    collection = chroma_client.get_collection("clinc150_kb")

    with open(f'{ARTIFACTS_PATH}/knowledge_base.pkl', 'rb') as f:
        knowledge_base = pickle.load(f)
    with open(f'{ARTIFACTS_PATH}/chunk_index.pkl', 'rb') as f:
        chunk_index = pickle.load(f)

    print("✅ All artifacts loaded successfully!")
    return tfidf, clf, le, intent_list, embedder, faiss_index, collection, knowledge_base, chunk_index

tfidf, clf, le, intent_list, embedder, faiss_index, collection, knowledge_base, chunk_index = load_artifacts()

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
        top5_idx = np.argsort(proba)[::-1][:5]
        top5 = {le.inverse_transform([i])[0]: float(proba[i]) for i in top5_idx}
    else:
        conf = 0.85
        top5 = {str(pred): 0.85}
    return str(pred), conf, top5

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
        ) if where else collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
    except:
        result = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
        filter_mode = "fallback"

    docs = []
    for doc_id, doc_text, meta, dist in zip(result['ids'][0], result['documents'][0], result['metadatas'][0], result['distances'][0]):
        docs.append({
            "title": meta["title"],
            "domain": meta["domain"],
            "resolution": knowledge_base.get(meta.get("intent"), {}).get("resolution", doc_text),
            "policy": knowledge_base.get(meta.get("intent"), {}).get("policy", ""),
            "retrieval_score": float(1 - dist),
            "filter_mode": filter_mode
        })
    return docs, filter_mode

def generate_with_groq(query: str, retrieved_docs: List[Dict], groq_key: str):
    if not groq_key or groq_key.strip() == "":
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Please contact customer support."
        return fallback, False

    system_prompt = """You are an AI customer support copilot for a bank. 
Generate accurate, concise (3-5 sentences), empathetic replies.
Use ONLY the provided documents. No hallucination.
For fraud/security issues: emphasize urgency and 24/7 helpline.
Plain text only. Start directly with the solution."""

    context = "\n\n".join([f"[Doc {i+1}: {d['title']} — {d['domain']}]\n{d['resolution']}" 
                          for i, d in enumerate(retrieved_docs)])

    prompt = f"{system_prompt}\n\nKNOWLEDGE BASE:\n{context}\n\nCUSTOMER QUERY: {query}\n\nSuggested Reply:"

    try:
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip(), True
    except Exception as e:
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Support will assist you shortly."
        return f"[Error] {fallback}", False

def evaluate_quality(generated: str, retrieved_docs: List[Dict]):
    rouge = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
    ref = retrieved_docs[0]['resolution'] if retrieved_docs else ""
    scores = rouge.score(ref, generated)
    ref_words = set(re.findall(r'\b\w+\b', ref.lower()))
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    faith = len(ref_words & gen_words) / max(len(ref_words), 1)

    return {
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3),
        "Faithfulness": round(faith, 3),
        "Words": len(generated.split()),
        "Quality": "High" if faith > 0.4 else "Medium" if faith > 0.2 else "Low"
    }

# ========================== GRADIO INTERFACE ==========================
def process_query(query, groq_key, top_k, use_filter):
    if not query or not query.strip():
        return "⚠️ Please enter a customer query", None, None, None, None

    start = time.time()
    pred_intent, confidence, top5 = predict_intent(query)
    retrieved_docs, filter_mode = retrieve_documents(query, pred_intent, confidence, int(top_k), use_filter)
    llm_response, success = generate_with_groq(query, retrieved_docs, groq_key)
    quality = evaluate_quality(llm_response, retrieved_docs)
    total_time = round((time.time() - start) * 1000)

    result = {
        "query": query,
        "predicted_intent": pred_intent,
        "confidence": confidence,
        "filter_mode": filter_mode,
        "retrieved_docs": retrieved_docs,
        "llm_response": llm_response,
        "quality": quality,
        "timings": total_time
    }

    status_msg = f"✅ Pipeline completed in **{total_time}ms** | Intent: **{pred_intent}** ({confidence:.1%}) | Filter: {filter_mode}"
    return status_msg, result, retrieved_docs, llm_response, quality

def approve_response(current_result, edited_response, history):
    if not current_result:
        return history, "Nothing to approve"
    entry = {**current_result, "final_response": edited_response, "status": "approved"}
    history.append(entry)
    return history, "✅ Response Approved & Saved!"

def reject_response(current_result, history):
    if not current_result:
        return history, "Nothing to reject"
    entry = {**current_result, "final_response": "", "status": "rejected"}
    history.append(entry)
    return history, "❌ Response Rejected"

with gr.Blocks(title="AI Support Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Customer Support Copilot\n**Week 3 • Gradio + Groq (Llama-3.1-70B)**")

    with gr.Tab("💬 Live Copilot"):
        groq_key = gr.Textbox(label="🔑 Groq API Key (Get free at groq.com)", type="password", placeholder="gsk_...")

        with gr.Row():
            query_input = gr.Textbox(label="Customer Query", placeholder="My payment failed but the money was deducted...", lines=3)
            with gr.Column(scale=1):
                top_k = gr.Slider(1, 5, value=3, step=1, label="Top-K Documents")
                use_filter = gr.Checkbox(value=True, label="Use Intent/Domain Filter")

        run_btn = gr.Button("🚀 Run AI Pipeline", variant="primary", size="large")

        status = gr.Markdown("Ready to process queries...")
        current_result_state = gr.State(None)
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📌 Predicted Intent")
                intent_display = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("### 📚 Retrieved Documents")
                docs_display = gr.JSON(label="Retrieved Documents")

        response_output = gr.Textbox(label="✍️ AI Suggested Reply (You can edit)", lines=8)

        quality_display = gr.JSON(label="📊 Response Quality")

        with gr.Row():
            approve_btn = gr.Button("✅ Approve & Send", variant="primary")
            reject_btn = gr.Button("❌ Reject")

        # Sample Queries
        with gr.Row():
            for sample in ["Payment failed but money deducted", "Account is blocked", "Unauthorized transaction", "How to check balance?"]:
                gr.Button(sample).click(lambda s=sample: s, outputs=query_input)

        # Button Actions
        run_btn.click(
            process_query,
            inputs=[query_input, groq_key, top_k, use_filter],
            outputs=[status, current_result_state, docs_display, response_output, quality_display]
        )

        approve_btn.click(
            approve_response,
            inputs=[current_result_state, response_output, history_state],
            outputs=[history_state, status]
        )

        reject_btn.click(
            reject_response,
            inputs=[current_result_state, history_state],
            outputs=[history_state, status]
        )

    with gr.Tab("📖 History"):
        gr.Markdown("Conversation history will appear here (can be extended)")

    gr.Markdown("---\n**Tip:** Get your free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)")

demo.launch(share=True)   # share=True gives public link (great for Colab)
