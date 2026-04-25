============================================================
  Loading AI Copilot components...
============================================================
  [OK] ML artifacts: LinearSVC, 150 intents
Loading weights: 100%
 103/103 [00:00<00:00, 358.76it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
ERROR:chromadb.telemetry.product.posthog:Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
  [OK] Embedder: all-MiniLM-L6-v2
  [OK] FAISS index: 124 vectors
  [OK] Chroma: 124 documents
  [OK] Groq client: model=llama-3.1-8b-instant
============================================================
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_2361/332312296.py in <cell line: 0>()
    876             with gr.Row():
    877                 sample_btns = [
--> 878                     gr.Button(label=q[:35]+"…" if len(q)>35 else q, size="sm")
    879                     for q in SAMPLE_QUERIES[:4]
    880                 ]

/usr/local/lib/python3.12/dist-packages/gradio/component_meta.py in wrapper(*args, **kwargs)
    187             return None
    188         else:
--> 189             return fn(self, **kwargs)
    190 
    191     return wrapper

TypeError: Button.__init__() got an unexpected keyword argument 'label'
