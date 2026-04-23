# =============================================================================
# WEEK 3 — AI CUSTOMER SUPPORT COPILOT (GRADIO + GROQ)
# Replaces Streamlit | Uses Groq (llama-3.1-70b-versatile) — fast & generous free tier
# Works in Colab / Local Laptop | No tunneling needed
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

# RAG components
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from rouge_score import rouge_scorer

# ========================== CONFIG ==========================
ARTIFACTS_PATH = "/content/drive/MyDrive/saved_models/artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# ========================== LOAD ARTIFACTS ==========================
@st.cache_resource  # No, this is Gradio, we'll use global
def load_artifacts():
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
        top5 = {pred: 0.85}
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
            "resolution": knowledge_base.get(meta["intent"], {}).get("resolution", doc_text),
            "policy": knowledge_base.get(meta["intent"], {}).get("policy", ""),
            "retrieval_score": float(1 - dist),
            "filter_mode": filter_mode
        })
    return docs, filter_mode

def generate_with_groq(query: str, retrieved_docs: List[Dict], groq_key: str):
    if not groq_key:
        return retrieved_docs[0]['resolution'] if retrieved_docs else "Please contact support.", False

    system_prompt = """You are an AI customer support copilot. 
Generate accurate, concise (3-5 sentences), empathetic replies.
Use ONLY the provided documents. No hallucination.
For fraud/security: urge immediate action + 24/7 helpline.
Plain text only. Start directly with solution."""

    context = "\n\n".join([f"[Doc {i+1}: {d['title']} — {d['domain']}]\n{d['resolution']}" 
                          for i, d in enumerate(retrieved_docs)])

    prompt = f"{system_prompt}\n\nKNOWLEDGE:\n{context}\n\nQUERY: {query}\n\nSuggested Reply:"

    try:
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip(), True
    except Exception as e:
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Support team will contact you shortly."
        return f"[Groq Error] {fallback}", False

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

# ========================== GRADIO APP ==========================
def process_query(query, groq_key, top_k, use_filter):
    if not query.strip():
        return "Please enter a query", None, None, None, None

    start = time.time()
    pred_intent, confidence, top5 = predict_intent(query)
    retrieved_docs, filter_mode = retrieve_documents(query, pred_intent, confidence, top_k, use_filter)
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

    return (
        f"✅ Done in {total_time}ms | Intent: {pred_intent} ({confidence:.1%})",
        result,
        retrieved_docs,
        llm_response,
        quality
    )

def approve(result, edited_response, history):
    if not result:
        return history, "Nothing to approve"
    entry = {**result, "final_response": edited_response, "status": "approved"}
    history.append(entry)
    return history, "✅ Approved!"

def reject(result, history):
    if not result:
        return history, "Nothing to reject"
    entry = {**result, "final_response": "", "status": "rejected"}
    history.append(entry)
    return history, "❌ Rejected"

with gr.Blocks(title="AI Support Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Customer Support Copilot\n**Week 3 • Gradio + Groq (llama-3.1-70b)**")

    with gr.Tab("💬 Live Copilot"):
        groq_key = gr.Textbox(label="Groq API Key (free at groq.com)", type="password", placeholder="gsk_...")
        
        with gr.Row():
            query = gr.Textbox(label="Customer Query", placeholder="My payment failed but money was deducted...", lines=3)
            with gr.Column():
                top_k = gr.Slider(1, 5, value=3, label="Top-K Docs")
                use_filter = gr.Checkbox(value=True, label="Use Intent Filter")

        run_btn = gr.Button("🚀 Analyse & Generate Reply", variant="primary")

        status = gr.Markdown()
        current_result = gr.State(None)
        history = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Predicted Intent**")
                intent_out = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("**Retrieved Documents**")
                docs_out = gr.JSON()

        response_box = gr.Textbox(label="Suggested Reply (Edit if needed)", lines=6)

        with gr.Row():
            quality_out = gr.JSON(label="Quality Metrics")
            gr.Markdown("**Pipeline Time**")
            time_out = gr.Markdown()

        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary")
            reject_btn = gr.Button("❌ Reject")

        # Sample buttons
        samples = ["My payment failed but money was deducted", 
                   "Account is blocked", 
                   "Someone made unauthorised transaction", 
                   "How to check balance?"]
        with gr.Row():
            for s in samples:
                gr.Button(s[:30]).click(lambda x=s: x, outputs=query)

        # Run pipeline
        run_btn.click(
            process_query,
            inputs=[query, groq_key, top_k, use_filter],
            outputs=[status, current_result, docs_out, response_box, quality_out]
        )

        approve_btn.click(approve, inputs=[current_result, response_box, history], outputs=[history, status])
        reject_btn.click(reject, inputs=[current_result, history], outputs=[history, status])

    with gr.Tab("📖 History"):
        gr.DataFrame(label="Conversation History")

    with gr.Tab("📊 Evaluation"):
        gr.Markdown("Evaluation dashboards from Week 2 will load here (CSV files)")

    gr.Markdown("---\n**Tip:** Get free Groq key at [console.groq.com](https://console.groq.com)")

demo.launch(share=True, server_name="0.0.0.0")   # share=True gives public link in Colab
