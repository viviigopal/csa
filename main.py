import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re
import os
from collections import Counter, defaultdict

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output folder for artifacts
os.makedirs('/content/drive/MyDrive/saved_models/artifacts', exist_ok=True)

print('All imports successful.')


# Load CLINC150 dataset from Hugging Face
# 'plus' variant includes out-of-scope (oos) examples at a higher density
raw = load_dataset('clinc_oos', 'plus')

print('Splits available:', list(raw.keys()))
print('Train features:', raw['train'].features)




def hf_split_to_df(split) -> pd.DataFrame:
    """Convert a Hugging Face dataset split to a pandas DataFrame."""
    df = pd.DataFrame({'text': split['text'], 'label': split['intent']})
    # Map integer labels to string intent names using the ClassLabel feature
    label_names = raw['train'].features['intent'].names
    df['intent_name'] = df['label'].map(lambda x: label_names[x])
    return df

train_df = hf_split_to_df(raw['train'])
val_df   = hf_split_to_df(raw['validation'])
test_df  = hf_split_to_df(raw['test'])

# Remove out-of-scope (oos) class — not a real intent
OOS_LABEL = 'oos'
train_df = train_df[train_df['intent_name'] != OOS_LABEL].reset_index(drop=True)
val_df   = val_df[val_df['intent_name']     != OOS_LABEL].reset_index(drop=True)
test_df  = test_df[test_df['intent_name']   != OOS_LABEL].reset_index(drop=True)

# Merge train + validation for maximum training data
train_full_df = pd.concat([train_df, val_df], ignore_index=True)

print(f'Train+Val size : {len(train_full_df):,}')
print(f'Test size      : {len(test_df):,}')
print(f'Unique intents : {train_full_df["intent_name"].nunique()}')

train_full_df.head()




# ── Domain mapping for CLINC150's 150 intents ──────────────────────────────
DOMAIN_MAP = {
    'Banking':        ['transfer', 'transactions', 'balance', 'freeze_account',
                       'pin_change', 'bill_due', 'pay_bill', 'bill_balance',
                       'interest_rate', 'routing', 'min_payment', 'order_checks',
                       'direct_deposit', 'credit_score', 'report_fraud',
                       'account_blocked', 'spending_history'],
    'Credit Cards':   ['credit_limit', 'credit_limit_change', 'damaged_card',
                       'replacement_card_duration', 'expiration_date',
                       'application_status', 'card_declined', 'apr'],
    'Travel':         ['book_flight', 'book_hotel', 'car_rental', 'travel_alert',
                       'travel_suggestion', 'travel_notification', 'carry_on',
                       'timezone', 'exchange_rate', 'flight_status',
                       'international_fees', 'lost_luggage', 'plug_type',
                       'vaccines', 'change_speed', 'cancel_reservation'],
    'Utilities':      ['weather', 'calculator', 'definition', 'translate',
                       'meaning_of_life', 'time', 'alarm', 'timer',
                       'date', 'calendar', 'reminder', 'reminder_update',
                       'shopping_list', 'shopping_list_update', 'todo_list',
                       'todo_list_update', 'taxes', 'user_name'],
    'Home':           ['smart_home', 'play_music', 'next_song', 'volume',
                       'skip', 'pause_music', 'repeat', 'what_song',
                       'sync_device', 'update_playlist', 'playlist'],
    'Auto & Commute': ['traffic', 'directions', 'gas', 'gas_type',
                       'distance', 'mpg', 'oil_change_when', 'oil_change_how',
                       'jump_start', 'uber', 'schedule_maintenance',
                       'tire_pressure', 'tire_change', 'insurance'],
    'Small Talk':     ['greeting', 'goodbye', 'thank_you', 'tell_joke',
                       'are_you_a_bot', 'who_made_you', 'how_old_are_you',
                       'fun_fact', 'what_can_i_ask_you', 'what_are_your_hobbies',
                       'do_you_have_pets', 'where_are_you_from', 'who_do_you_work_for',
                       'change_ai_name', 'change_user_name', 'carry_on'],
    'Meta':           ['iot_cleaning', 'iot_coffee', 'iot_hue_lightchange',
                       'iot_hue_lighton', 'iot_hue_lightoff', 'iot_hue_lightdim',
                       'iot_hue_lightup', 'iot_wemo_on', 'iot_wemo_off',
                       'iot_cleaning'],
    'Shopping':       ['order_status', 'refund', 'where_are_you_from',
                       'products_and_services', 'return_policy', 'rewards_balance',
                       'meal_suggestion', 'restaurant_suggestion', 'food_last',
                       'ingredient_substitution', 'calories', 'nutrition_info',
                       'recipe', 'confirm_reservation', 'how_busy',
                       'accept_reservations', 'restaurant_reviews'],
    'General':        []   # catch-all
}

def get_domain(intent: str) -> str:
    for domain, intents in DOMAIN_MAP.items():
        if intent in intents:
            return domain
    return 'General'

train_full_df['domain'] = train_full_df['intent_name'].map(get_domain)
test_df['domain']       = test_df['intent_name'].map(get_domain)

print(train_full_df['domain'].value_counts())




def clean_text(text: str) -> str:
    """
    Minimal but effective text cleaning:
    - Lowercase
    - Remove punctuation EXCEPT apostrophes (preserves contractions like "can't")
    - Collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)   # keep letters, digits, apostrophe
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# Apply cleaning
train_full_df['clean_text'] = train_full_df['text'].map(clean_text)
test_df['clean_text']       = test_df['text'].map(clean_text)

"""# Before / after example
sample_idx = 56
print('Original :', train_full_df['text'].iloc[sample_idx])
print('Cleaned  :', train_full_df['clean_text'].iloc[sample_idx])"""




# TF-IDF with unigrams + bigrams, max 20k features
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20_000,
    sublinear_tf=True,     # apply log normalization to term frequencies
    min_df=2               # ignore terms appearing in fewer than 2 docs
)

X_train = tfidf.fit_transform(train_full_df['clean_text'])
X_test  = tfidf.transform(test_df['clean_text'])

y_train = train_full_df['intent_name'].values
y_test  = test_df['intent_name'].values

print(f'X_train shape : {X_train.shape}')
print(f'X_test  shape : {X_test.shape}')
print(f'Vocabulary size: {len(tfidf.vocabulary_):,}')




# Print top TF-IDF features per intent (sample of 10 intents)
feature_names = np.array(tfidf.get_feature_names_out())
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Refit a simple LogReg just for feature inspection
_lr_inspect = LogisticRegression(C=5, max_iter=1000, random_state=RANDOM_STATE)
_lr_inspect.fit(X_train, y_train)

SAMPLE_INTENTS = ['transfer', 'book_flight', 'weather', 'credit_limit',
                  'cancel_reservation', 'balance', 'tell_joke', 'refund',
                  'alarm', 'directions']

print('Top 10 TF-IDF features per intent:')
print('=' * 55)
classes = list(_lr_inspect.classes_)
for intent in SAMPLE_INTENTS:
    if intent not in classes:
        continue
    idx = classes.index(intent)
    top_idx = np.argsort(_lr_inspect.coef_[idx])[-10:][::-1]
    print(f'\n[{intent}]')
    print('  ', ', '.join(feature_names[top_idx]))




 """
    'Random Forest':       RandomForestClassifier(n_estimators=300,
                                                   max_depth=None,
                                                   n_jobs=-1,
                                                   random_state=RANDOM_STATE),
    'XGBoost':             XGBClassifier(n_estimators=300,
                                          learning_rate=0.1,
                                          max_depth=6,
                                          eval_metric='mlogloss',
                                          tree_method='hist',
                                          n_jobs=-1,
                                          random_state=RANDOM_STATE),"""




from sklearn.preprocessing import LabelEncoder

# XGBoost requires integer labels — encode them
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

models = {
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(C=5, max_iter=1000,
                                               solver='lbfgs',
                                               multi_class='multinomial',
                                               random_state=RANDOM_STATE),
    'Linear SVM':          LinearSVC(C=1.0, max_iter=2000,
                                     random_state=RANDOM_STATE),

}

results = {}

for name, model in models.items():
    print(f'Training {name}...', end=' ', flush=True)

    if name == 'XGBoost':
        # XGBoost needs integer labels
        model.fit(X_train, y_train_enc)
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)   # decode back to intent strings
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec,
                     'y_pred': y_pred, 'model': model}
    print(f'Accuracy={acc:.4f}  F1={f1:.4f}')

# Also store the label encoder for use in Step 10 artifacts
results['_label_encoder'] = le
print('\nAll models trained.')




best_model_obj = results[best_model_name]['model']

# Save TF-IDF vectorizer
with open('/content/drive/MyDrive/saved_models/artifacts/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save best model
with open('/content/drive/MyDrive/saved_models/artifacts/best_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

# Save intent list
intent_list = sorted(train_full_df['intent_name'].unique().tolist())
with open('/content/drive/MyDrive/saved_models/artifacts/intent_list.pkl', 'wb') as f:
    pickle.dump(intent_list, f)
with open('/content/drive/MyDrive/saved_models/artifacts/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save cleaned datasets for RAG pipeline
train_full_df.to_csv('/content/drive/MyDrive/saved_models/artifacts/train_cleaned.csv', index=False)
test_df.to_csv('/content/drive/MyDrive/saved_models/artifacts/test_cleaned.csv', index=False)

print('Saved artifacts:')
for fname in os.listdir('artifacts'):
    fpath = os.path.join('artifacts', fname)
    size  = os.path.getsize(fpath) / 1024
    print(f'  {fname:40s}  {size:8.1f} KB')




# =============================================================================
# WEEK 2 — RAG PIPELINE FOR AI CUSTOMER SUPPORT COPILOT
# Dataset : CLINC150 (same as Week 1)
# Pipeline : Knowledge Base → Embeddings → Vector DB → Retrieval → LLM → Eval
# =============================================================================

# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import pickle
import textwrap
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Sentence Transformers (embeddings) ────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Vector stores ─────────────────────────────────────────────────────────
import faiss
import chromadb
from chromadb.config import Settings

# ── Evaluation ────────────────────────────────────────────────────────────
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ── Week 1 artifacts ──────────────────────────────────────────────────────
# (tfidf_vectorizer.pkl, best_model.pkl, label_encoder.pkl must be in ./artifacts/)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('/content/drive/MyDrive/saved_models/artifacts', exist_ok=True)
os.makedirs('/content/drive/MyDrive/saved_models/artifacts/chroma_db', exist_ok=True)

print("=" * 65)
print(" WEEK 2 — RAG PIPELINE")
print("=" * 65)

# =============================================================================
# STEP 1 — LOAD WEEK 1 ARTIFACTS
# We reuse the trained classifier, TF-IDF vectoriser, and label encoder
# from Week 1 to predict intent → pre-filter vector search.
# =============================================================================
print("\n[1/8] Loading Week 1 artifacts ...")

with open('/content/drive/MyDrive/saved_models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
  tfidf = pickle.load(f)



with open('/content/drive/MyDrive/saved_models/artifacts/best_model.pkl', 'rb') as f:
  clf = pickle.load(f)

with open('/content/drive/MyDrive/saved_models/artifacts/label_encoder.pkl', 'rb') as f:
  le = pickle.load(f)

with open('/content/drive/MyDrive/saved_models/artifacts/intent_list.pkl', 'rb') as f:
  intent_list = pickle.load(f)

train_df = pd.read_csv('/content/drive/MyDrive/saved_models/artifacts/train_cleaned.csv')
test_df = pd.read_csv('/content/drive/MyDrive/saved_models/artifacts/test_cleaned.csv')

print(f" Classifier : {type(clf).__name__}")
print(f" Intents loaded : {len(intent_list)}")
print(f" Test samples : {len(test_df)}")




def build_document_chunks(knowledge_base: dict) -> List[Dict]:

  chunks = []
  for intent_name, doc in knowledge_base.items():
  # Combine all fields into a rich text chunk
    text = (
            f"Intent: {intent_name}\n"
            f"Title: {doc['title']}\n"
            f"Domain: {doc['domain']}\n"
            f"Description: {doc['description']}\n"
            f"Common questions: {' | '.join(doc['sample_queries'])}\n"
            f"Resolution: {doc['resolution']}\n"
            f"Policy: {doc['policy']}"
            )
    chunks.append({
          "chunk_id": f"chunk_{intent_name}",
          "intent": intent_name,
          "domain": doc["domain"],
          "title": doc["title"],
          "text": text,
          "resolution": doc["resolution"],
          "policy": doc.get("policy", ""),
          })
  return chunks

chunks = build_document_chunks(KNOWLEDGE_BASE)
print(f" Total chunks created: {len(chunks)}")
print(f" Avg chunk length : {np.mean([len(c['text']) for c in chunks]):.0f} chars")

# Save knowledge base
with open('/content/drive/MyDrive/saved_models/artifacts/knowledge_base.pkl', 'wb') as f:
  pickle.dump(KNOWLEDGE_BASE, f)
with open('/content/drive/MyDrive/saved_models/artifacts/chunks.pkl', 'wb') as f:
  pickle.dump(chunks, f)

print(" Knowledge base and chunks saved.")





# =============================================================================
# STEP 3 — CREATE EMBEDDINGS

# =============================================================================
print("\n[3/8] Creating embeddings ...")

# ── EMBEDDING MODEL COMPARISON ────────────────────────────────────────────
EMBEDDING_MODELS = {
"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
"all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}


SELECTED_MODEL = "all-MiniLM-L6-v2"

print(f" Loading embedding model: {SELECTED_MODEL}")
embedder = SentenceTransformer(SELECTED_MODEL)

# Embed all document chunks
chunk_texts = [c["text"] for c in chunks]
print(f" Embedding {len(chunk_texts)} chunks ...")
chunk_embeddings = embedder.encode(
chunk_texts,
batch_size=64,
show_progress_bar=True,
normalize_embeddings=True
)
print(f" Embedding shape: {chunk_embeddings.shape}") # (n_chunks, 384)

# Save embeddings
np.save('/content/drive/MyDrive/saved_models/artifacts/chunk_embeddings.npy', chunk_embeddings)
print(" Embeddings saved.")




# =============================================================================
# STEP 4 — BUILD VECTOR STORES (FAISS + CHROMA)

# =============================================================================
print("\n[4/8] Building vector stores ...")

# ── 4a: FAISS ─────────────────────────────────────────────────────────────
dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim) # Inner Product = cosine if normalised
faiss_index.add(chunk_embeddings.astype(np.float32))
faiss.write_index(faiss_index, '/content/drive/MyDrive/saved_models/artifacts/faiss_index.bin')
print(f" FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")

# ── 4b: CHROMA ────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path='/content/drive/MyDrive/saved_models/artifacts/chroma_db')

# Delete existing collection if re-running
try:
  chroma_client.delete_collection("clinc150_kb")
except:
  pass

collection = chroma_client.create_collection(
name="clinc150_kb",
metadata={"hnsw:space": "cosine"}
)

collection.add(
ids=[c["chunk_id"] for c in chunks],
embeddings=chunk_embeddings.tolist(),
documents=[c["text"] for c in chunks],
metadatas=[{
"intent": c["intent"],
"domain": c["domain"],
"title": c["title"]
} for c in chunks]
)
print(f" Chroma collection built: {collection.count()} documents")

# Save chunk index (for FAISS result lookup)
chunk_index = {i: c for i, c in enumerate(chunks)}
with open('/content/drive/MyDrive/saved_models/artifacts/chunk_index.pkl', 'wb') as f:
  pickle.dump(chunk_index, f)






# =============================================================================
# STEP 5 — RETRIEVAL FUNCTIONS
# =============================================================================
print("\n[5/8] Setting up retrieval pipeline ...")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def predict_intent(query: str) -> Tuple[str, float]:
    """
    Use Week 1 TF-IDF + classifier to predict intent and confidence.
    Returns (intent_name, confidence_score).
    """
    clean = clean_text(query)
    vec = tfidf.transform([clean])
    pred_enc = clf.predict(vec)[0]

    # Get probability if available
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(vec)[0]
        confidence = float(np.max(proba))

    elif hasattr(clf, 'decision_function'):
        # Normalize decision function scores via softmax
        scores = clf.decision_function(vec)[0]
        exp_s = np.exp(scores - np.max(scores))
        proba = exp_s / exp_s.sum()
        confidence = float(np.max(proba))

    else:
        confidence = 0.8

    intent = pred_enc  # already string
    return intent, confidence


def retrieve_faiss(query: str, top_k: int = 3) -> List[Dict]:
    """Full FAISS retrieval — no intent filter."""
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunk_index[idx].copy()
        chunk['retrieval_score'] = float(score)
        results.append(chunk)

    return results


def retrieve_chroma(
    query: str,
    predicted_intent: str,
    confidence: float,
    top_k: int = 3
) -> List[Dict]:
    """
    Chroma retrieval with confidence-based pre-filtering:
    - confidence >= 0.70 → intent filter
    - 0.40–0.70 → domain filter
    - < 0.40 → no filter
    """

    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    # Determine domain
    predicted_domain = KNOWLEDGE_BASE.get(predicted_intent, {}).get("domain", "General")

    if confidence >= 0.70 and predicted_intent in KNOWLEDGE_BASE:
        where_filter = {"intent": {"$eq": predicted_intent}}
        filter_mode = "intent"

    elif confidence >= 0.40:
        where_filter = {"domain": {"$eq": predicted_domain}}
        filter_mode = "domain"

    else:
        where_filter = None
        filter_mode = "none"

    try:
        if where_filter:
            result = collection.query(
                query_embeddings=q_emb,
                n_results=min(top_k, collection.count()),
                where=where_filter
            )
        else:
            result = collection.query(
                query_embeddings=q_emb,
                n_results=top_k
            )

    except Exception:
        # fallback
        result = collection.query(query_embeddings=q_emb, n_results=top_k)
        filter_mode = "fallback"

    results = []
    for doc_id, doc, meta, dist in zip(
        result['ids'][0],
        result['documents'][0],
        result['metadatas'][0],
        result['distances'][0]
    ):
        results.append({
            "chunk_id": doc_id,
            "intent": meta["intent"],
            "domain": meta["domain"],
            "title": meta["title"],
            "text": doc,
            "resolution": KNOWLEDGE_BASE.get(meta["intent"], {}).get("resolution", doc),
            "retrieval_score": float(1 - dist),
            "filter_mode": filter_mode
        })

    return results


print(" FAISS and Chroma retrieval functions ready.")
print(" Confidence routing: >=0.70 → intent | 0.40–0.70 → domain | <0.40 → none")




# =============================================================================
# STEP 6 — LLM RESPONSE GENERATION (GEMINI)
# =============================================================================
print("\n[6/8] Setting up LLM response generation ...")

import google.generativeai as genai

# Set API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDG4FuT7WswJRLoj1hWWiaTEDL-Vh14ONg")
genai.configure(api_key=GEMINI_API_KEY)

# Load model
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI customer support copilot for a bank and financial services company.
Your role is to help support agents by generating accurate, concise, and empathetic suggested replies.

INSTRUCTIONS:
- Use ONLY the information provided in the retrieved knowledge base documents below.
- If the documents do not contain enough information, say so honestly — do not guess.
- Keep replies concise (3–5 sentences max) and professional.
- Address the customer's specific issue directly.
- If the issue involves account security or fraud, always prioritize urgency and direct to 24/7 helpline.
- End with an offer for further assistance.

FORMAT:
- Respond in plain text — no markdown, no bullet points.
- Begin directly with the resolution — no filler phrases.
- Use "you" and "your" to address the customer directly.
"""


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build the user message with retrieved context.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Document {i}: {chunk['title']} — {chunk['domain']}]\n{chunk['resolution']}"
        )

    context_str = "\n\n".join(context_parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"RETRIEVED KNOWLEDGE BASE DOCUMENTS:\n"
        f"{context_str}\n\n"
        f"CUSTOMER QUERY:\n{query}\n\n"
        f"Generate a helpful suggested reply for the support agent to send to the customer."
    )


def generate_response(
    query: str,
    retrieved_chunks: List[Dict],
    model_name: str = "gemini-2.5-flash",
    max_tokens: int = 300
) -> Dict:
    """
    Generate LLM response using Gemini.
    """
    prompt = build_prompt(query, retrieved_chunks)

    try:
        response = llm_model.generate_content(prompt)

        return {
            "response": response.text,
            "model": model_name,
            "input_tokens": 0,   # Gemini doesn't always expose token usage easily
            "output_tokens": 0,
            "success": True
        }

    except Exception as e:
        return {
            "response": f"[LLM Error: {str(e)}] Fallback: {retrieved_chunks[0]['resolution'] if retrieved_chunks else 'No documents retrieved.'}",
            "model": model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "success": False
        }


def full_pipeline(query: str, use_chroma: bool = True, top_k: int = 3) -> Dict:
    """
    End-to-end pipeline:
    Query → Intent → Retrieval → LLM → Response
    """

    # Step 1: Predict intent
    predicted_intent, confidence = predict_intent(query)

    # Step 2: Retrieve documents
    if use_chroma:
        retrieved = retrieve_chroma(query, predicted_intent, confidence, top_k=top_k)
        retrieval_method = f"Chroma (filter={retrieved[0]['filter_mode'] if retrieved else 'none'})"
    else:
        retrieved = retrieve_faiss(query, top_k=top_k)
        retrieval_method = "FAISS (no filter)"

    # Step 3: Generate response
    llm_result = generate_response(query, retrieved)

    return {
        "query": query,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "retrieval_method": retrieval_method,
        "retrieved_docs": retrieved,
        "llm_response": llm_result["response"],
        "llm_success": llm_result["success"],
        "model_used": llm_result["model"],
    }


print(" Gemini LLM pipeline ready. Model: gemini-2.5-flash")
print(" NOTE: Set GEMINI_API_KEY environment variable before running.")




# =============================================================================
# STEP 7 — EVALUATION
# =============================================================================
print("\n[7/8] Running evaluation ...")


# ── 7a: RETRIEVAL EVALUATION ──────────────────────────────────────────────
def recall_at_k(predicted_intents: List[str], true_intent: str, k: int) -> float:
    """Recall@K"""
    return float(true_intent in predicted_intents[:k])


def reciprocal_rank(predicted_intents: List[str], true_intent: str) -> float:
    """MRR"""
    for i, intent in enumerate(predicted_intents):
        if intent == true_intent:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(predicted_intents: List[str], true_intent: str, k: int) -> float:
    """NDCG@K"""
    dcg = 0.0
    for i, intent in enumerate(predicted_intents[:k]):
        if intent == true_intent:
            dcg += 1.0 / np.log2(i + 2)

    idcg = 1.0
    return dcg / idcg if idcg > 0 else 0.0


# ── Build eval dataset ────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 200
np.random.seed(RANDOM_STATE)

eval_samples = (
    test_df.groupby('intent_name', group_keys=False)
    .apply(lambda x: x.sample(min(2, len(x)), random_state=RANDOM_STATE))
    .reset_index(drop=True)
    .head(EVAL_SAMPLE_SIZE)
)

print(f" Retrieval eval on {len(eval_samples)} test queries ...")

retrieval_results = []

for _, row in eval_samples.iterrows():
    query = row['text']
    true_int = row['intent_name']

    # FAISS
    faiss_docs = retrieve_faiss(query, top_k=5)
    faiss_ints = [d['intent'] for d in faiss_docs]

    # Chroma
    pred_intent, conf = predict_intent(query)
    chroma_docs = retrieve_chroma(query, pred_intent, conf, top_k=5)
    chroma_ints = [d['intent'] for d in chroma_docs]

    retrieval_results.append({
        "query": query,
        "true_intent": true_int,
        "pred_intent": pred_intent,
        "confidence": conf,
        "faiss_r@1": recall_at_k(faiss_ints, true_int, 1),
        "faiss_r@3": recall_at_k(faiss_ints, true_int, 3),
        "faiss_r@5": recall_at_k(faiss_ints, true_int, 5),
        "faiss_mrr": reciprocal_rank(faiss_ints, true_int),
        "faiss_ndcg@3": ndcg_at_k(faiss_ints, true_int, 3),
        "chroma_r@1": recall_at_k(chroma_ints, true_int, 1),
        "chroma_r@3": recall_at_k(chroma_ints, true_int, 3),
        "chroma_r@5": recall_at_k(chroma_ints, true_int, 5),
        "chroma_mrr": reciprocal_rank(chroma_ints, true_int),
        "chroma_ndcg@3": ndcg_at_k(chroma_ints, true_int, 3),
    })

ret_df = pd.DataFrame(retrieval_results)

print("\n=== RETRIEVAL EVALUATION RESULTS ===")

ret_summary = pd.DataFrame({
    "Metric": ["Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@3"],
    "FAISS (no filter)": [
        ret_df['faiss_r@1'].mean(),
        ret_df['faiss_r@3'].mean(),
        ret_df['faiss_r@5'].mean(),
        ret_df['faiss_mrr'].mean(),
        ret_df['faiss_ndcg@3'].mean(),
    ],
    "Chroma (intent filter)": [
        ret_df['chroma_r@1'].mean(),
        ret_df['chroma_r@3'].mean(),
        ret_df['chroma_r@5'].mean(),
        ret_df['chroma_mrr'].mean(),
        ret_df['chroma_ndcg@3'].mean(),
    ]
})

print(ret_summary.to_string(index=False, float_format='{:.4f}'.format))

ret_summary.to_csv('/content/drive/MyDrive/saved_models/artifacts/retrieval_eval_results.csv', index=False)
ret_df.to_csv('/content/drive/MyDrive/saved_models/artifacts/retrieval_eval_detailed.csv', index=False)







# ── 7b: GENERATION EVALUATION ─────────────────────────────────────────────
print("\n Running generation evaluation (30 samples) ...")

GEN_EVAL_SIZE = 30
gen_samples = eval_samples.head(GEN_EVAL_SIZE)

rouge = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

gen_results = []

# Gemini key check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDG4FuT7WswJRLoj1hWWiaTEDL-Vh14ONg")

for _, row in gen_samples.iterrows():
    query = row['text']
    true_int = row['intent_name']

    pred_intent, conf = predict_intent(query)
    retrieved = retrieve_chroma(query, pred_intent, conf, top_k=3)

    reference = KNOWLEDGE_BASE.get(true_int, {}).get("resolution", "")
    retrieved_text = retrieved[0]['resolution'] if retrieved else ""

    # Gemini generation
    if GEMINI_API_KEY != "AIzaSyDG4FuT7WswJRLoj1hWWiaTEDL-Vh14ONg":
        llm_res = generate_response(query, retrieved)
        generated = llm_res["response"]
    else:
        generated = retrieved_text

    scores = rouge.score(reference, generated)
    rougeL = scores['rougeL'].fmeasure
    rouge1 = scores['rouge1'].fmeasure

    ref_words = set(re.findall(r'\b\w+\b', retrieved_text.lower()))
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    faithfulness = len(ref_words & gen_words) / max(len(ref_words), 1)

    gen_results.append({
        "query": query,
        "true_intent": true_int,
        "pred_intent": pred_intent,
        "confidence": conf,
        "generated": generated,
        "reference": reference,
        "rougeL": rougeL,
        "rouge1": rouge1,
        "faithfulness": faithfulness,
    })

gen_df = pd.DataFrame(gen_results)
gen_df.to_csv('/content/drive/MyDrive/saved_models/artifacts/generation_eval_results.csv', index=False)

print("\n=== GENERATION EVALUATION RESULTS ===")
print(f" ROUGE-L (mean): {gen_df['rougeL'].mean():.4f}")
print(f" ROUGE-1 (mean): {gen_df['rouge1'].mean():.4f}")
print(f" Faithfulness (mean): {gen_df['faithfulness'].mean():.4f}")


# ── 7c: ABLATION ──────────────────────────────────────────────────────────
print("\n Ablation: intent filter ON vs OFF ...")

ablation_results = []

for _, row in eval_samples.head(100).iterrows():
    query = row['text']
    true_int = row['intent_name']

    pred_int, conf = predict_intent(query)

    chroma_filt = retrieve_chroma(query, pred_int, conf, top_k=3)
    r3_filt = recall_at_k([d['intent'] for d in chroma_filt], true_int, 3)

    chroma_nofilt = retrieve_chroma(query, pred_int, 0.0, top_k=3)
    r3_nofilt = recall_at_k([d['intent'] for d in chroma_nofilt], true_int, 3)

    ablation_results.append({
        "with_filter": r3_filt,
        "without_filter": r3_nofilt
    })

abl_df = pd.DataFrame(ablation_results)

print(f" Recall@3 WITH filter: {abl_df['with_filter'].mean():.4f}")
print(f" Recall@3 WITHOUT filter: {abl_df['without_filter'].mean():.4f}")
print(f" Delta: {(abl_df['with_filter'].mean() - abl_df['without_filter'].mean()):.4f}")

abl_df.to_csv('/content/drive/MyDrive/saved_models/artifacts/ablation_results.csv', index=False)




def retrieve_documents(
    query: str, predicted_intent: str, confidence: float,
    embedder, faiss_index, collection, knowledge_base, chunk_index,
    top_k: int = 3, use_filter: bool = True
) -> Tuple[List[Dict], str]:
    """
    Retrieve relevant docs with confidence-based routing.
    Returns (docs, filter_mode_description).
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)
    predicted_domain = knowledge_base.get(predicted_intent, {}).get("domain", "General")

    if use_filter:
        if confidence >= 0.70 and predicted_intent in knowledge_base:
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
            n_results=min(top_k, collection.count()),
            where=where
        ) if where else collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k
        )
    except Exception:
        result = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
        filter_mode = "fallback"

    docs = []
    for doc_id, doc, meta, dist in zip(
        result['ids'][0], result['documents'][0],
        result['metadatas'][0], result['distances'][0]
    ):
        docs.append({
            "chunk_id": doc_id,
            "intent": meta["intent"],
            "domain": meta["domain"],
            "title": meta["title"],
            "text": doc,
            "resolution": knowledge_base.get(meta["intent"], {}).get("resolution", doc),
            "policy": knowledge_base.get(meta["intent"], {}).get("policy", ""),
            "retrieval_score": float(1 - dist),
            "filter_mode": filter_mode,
        })
    return docs, filter_mode

def generate_with_gemini(query: str, retrieved_docs: List[Dict], api_key: str) -> Dict:
    """
    Generate response using Gemini 2.0 Flash (free tier).
    WHY GEMINI FLASH:
    - Free tier: 15 requests/min, 1M tokens/day
    - Fast: typically <2 second response
    - Strong context utilisation for RAG
    - No billing required
    """
    import google.generativeai as genai

    SYSTEM_PROMPT = """You are an AI customer support copilot for a bank and financial services company.
Generate accurate, concise (3-5 sentences), and empathetic suggested replies for support agents.
Use ONLY the retrieved documents below. Do not guess or hallucinate.
For fraud/security: emphasise urgency and 24/7 helpline.
Plain text only — no markdown, no bullet points. Start directly with the solution."""

    context = "\n\n".join([
        f"[Doc {i+1}: {doc['title']} — {doc['domain']}]\n{doc['resolution']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    prompt = f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nCUSTOMER QUERY: {query}\n\nSuggested reply:"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.3
            )
        )
        return {"response": resp.text.strip(), "success": True, "model": "gemini-2.5-flash"}
    except Exception as e:
        fallback = retrieved_docs[0]['resolution'] if retrieved_docs else "Please contact customer care."
        return {"response": fallback, "success": False, "error": str(e), "model": "fallback"}

def evaluate_response_quality(generated: str, retrieved_docs: List[Dict]) -> Dict:
    """
    Quick quality check on generated response:
    - ROUGE-L vs top retrieved document
    - Faithfulness (lexical overlap with retrieved context)
    - Length check
    """
    rouge = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
    reference = retrieved_docs[0]['resolution'] if retrieved_docs else ""
    scores = rouge.score(reference, generated)

    ref_words = set(re.findall(r'\b\w+\b', reference.lower()))
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    faithfulness = len(ref_words & gen_words) / max(len(ref_words), 1)

    word_count = len(generated.split())

    return {
        "rougeL": scores['rougeL'].fmeasure,
        "rouge1": scores['rouge1'].fmeasure,
        "faithfulness": faithfulness,
        "word_count": word_count,
        "quality_label": (
            "High" if faithfulness > 0.4 else
            "Medium" if faithfulness > 0.2 else "Low"
        )
    }




# ==============================
# SIMPLE DEMO RUN (NO UI)
# ==============================

def demo_run(query: str):
    print("\n" + "="*60)
    print("🔍 QUERY:")
    print(query)

    # Step 1 — Intent
    pred_intent, conf = predict_intent(query)
    print("\n🧠 PREDICTED INTENT:", pred_intent)
    print("📊 CONFIDENCE:", round(conf, 3))

    # Step 2 — Retrieval
    retrieved = retrieve_chroma(query, pred_intent, conf, top_k=3)
    print("\n📚 RETRIEVED DOCUMENTS:")

    for i, doc in enumerate(retrieved, 1):
        print(f"\n[{i}] {doc['title']} ({doc['domain']})")
        print("Score:", round(doc['retrieval_score'], 3))
        print("Filter:", doc.get("filter_mode", "N/A"))

    # Step 3 — LLM
    llm_result = generate_response(query, retrieved)

    print("\n" + "="*60)
    print("🤖 LLM RESPONSE:\n")
    print(llm_result["response"])
    print("="*60)

    return llm_result


# ==============================
# RUN MULTIPLE TEST QUERIES
# ==============================

queries = [
    "I lost my debit card, what should I do?",
    "My transaction failed but money got deducted",
    "How can I reset my net banking password?",
    "I see unauthorized transactions on my account",
]

for q in queries:
    demo_run(q)

