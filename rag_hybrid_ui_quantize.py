import os
import torch
import faiss
import pickle
import numpy as np
import gradio as gr
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gpt import GPTLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----- Configuration -----
CORPUS_DIR = "docs/"
CHUNK_SIZE = 300
STRIDE = 150  # 50% overlap
TOP_K = 3
ALPHA = 0.5  # hybrid weighting: 1 = pure embedding, 0 = pure tf-idf
EMBED_MODEL = 'all-MiniLM-L6-v2'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
USE_QUANTIZATION = True  # toggle quantization logic

# ----- Try to import bitsandbytes (optional) -----
_has_bnb = False
try:
    import bitsandbytes as bnb  # noqa: F401
    _has_bnb = True
except Exception:
    _has_bnb = False

# ----- Prepare BitsAndBytesConfig for 4-bit (safe for older GPUs) -----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",    # usually best quality/speed trade-off
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ----- Load the small HuggingFace model (SmolLM) with quantization when possible -----
print("Loading SmolLM tokenizer / model (quantization-aware)...")
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
smol_tokenizer = AutoTokenizer.from_pretrained(model_id)

smol_model = None
if USE_QUANTIZATION and CUDA and _has_bnb:
    try:
        print("[INFO] Attempting 4-bit load with bitsandbytes (recommended for older GPUs).")
        smol_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("[OK] SmolLM loaded in 4-bit via bitsandbytes.")
    except Exception as e:
        print("[WARN] 4-bit load failed:", repr(e))
        smol_model = None

if smol_model is None:
    # Fallback: try fp16 on CUDA, otherwise CPU fp32
    try:
        if CUDA:
            print("[INFO] Loading SmolLM in fp16 on CUDA as fallback.")
            smol_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Auto device_map should place it correctly; ensure model is on DEVICE
            try:
                smol_model.to(DEVICE)
            except Exception:
                pass
        else:
            print("[INFO] Loading SmolLM on CPU (fp32). If you want quantization, install bitsandbytes + CUDA.")
            smol_model = AutoModelForCausalLM.from_pretrained(model_id)
            smol_model.to(DEVICE)
    except Exception as e:
        print("[ERROR] Failed to load SmolLM model:", repr(e))
        raise

# ----- Load and Chunk Documents with Sliding Window -----
print("Loading and chunking documents with sliding window...")

def sliding_window_chunk(text, chunk_size=300, stride=150):
    # Keep original behavior but simpler: split by characters with overlap
    chunks = []
    n = len(text)
    for i in range(0, n, stride):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks

documents = []
if os.path.isdir(CORPUS_DIR):
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(CORPUS_DIR, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
else:
    print(f"[WARN] CORPUS_DIR {CORPUS_DIR} not found. No documents loaded.")

chunks = []
for doc in documents:
    chunks.extend(sliding_window_chunk(doc, chunk_size=CHUNK_SIZE, stride=STRIDE))

print(f"Loaded {len(documents)} documents with {len(chunks)} total chunks.")

# ----- Embed Chunks -----
print("Embedding chunks...")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
if len(chunks) > 0:
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
else:
    chunk_embeddings = np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)

# ----- Build FAISS Index -----
print("Building FAISS index...")
if chunk_embeddings.shape[0] > 0:
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
else:
    index = None

# ----- Build TF-IDF Index -----
print("Building TF-IDF index...")
if len(chunks) > 0:
    tfidf_vectorizer = TfidfVectorizer().fit(chunks)
    tfidf_matrix = tfidf_vectorizer.transform(chunks)
else:
    tfidf_vectorizer = None
    tfidf_matrix = None

# ----- Load GPT model from local checkpoint and apply dynamic quantization if possible -----
print("Loading custom GPT model (gpt.pth)...")
model = GPTLanguageModel()
map_loc = "cpu" if (USE_QUANTIZATION and not CUDA) else DEVICE

# load state dict safely
state = torch.load('gpt.pth', map_location=map_loc)
try:
    model.load_state_dict(state)
except Exception as e:
    # if checkpoint contains additional wrapper keys, try common fallbacks
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        print("[WARN] load_state_dict failed, attempting direct assignment if shapes match:", repr(e))
        # give up re-raising; allow the user to debug
        raise

# If using dynamic quantization, it's applied only on CPU
if USE_QUANTIZATION and not CUDA:
    try:
        print("[INFO] Applying dynamic quantization to GPTLanguageModel (CPU)...")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.to("cpu")
        print("[OK] Dynamic quantization applied to GPTLanguageModel.")
    except Exception as e:
        print("[WARN] Dynamic quantization failed:", repr(e))
        model.to(map_loc)
else:
    # GPU path or no quantization requested
    model.to(DEVICE)

model.eval()

# ----- Load Vocab -----
with open('vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ----- Hybrid Retrieval Function (with TF-IDF Rare Term Boosting) -----
def hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA, rare_term_boost=True):
    if index is None or tfidf_vectorizer is None:
        return []
    query_embedding = embedder.encode([prompt])
    D_embed, I_embed = index.search(np.array(query_embedding), top_k * 2)

    tfidf_query = tfidf_vectorizer.transform([prompt])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]
    I_tfidf = np.argsort(tfidf_scores)[::-1][:top_k * 2]

    scores = {}
    for i in I_embed[0]:
        # D_embed holds distances; convert to similarity-like score
        try:
            pos = int(list(I_embed[0]).index(i))
            dist = float(D_embed[0][pos])
            scores[i] = scores.get(i, 0) + alpha * (1.0 / (1.0 + dist))
        except Exception:
            continue
    for i in I_tfidf:
        scores[i] = scores.get(i, 0) + (1 - alpha) * float(tfidf_scores[i])

    if rare_term_boost:
        tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
        tfidf_query_vector = tfidf_query.toarray()[0]
        rare_terms = [tfidf_vocab[i] for i in np.argsort(tfidf_query_vector)[::-1][:3] if tfidf_query_vector[i] > 0]
        for idx, chunk in enumerate(chunks):
            for term in rare_terms:
                if term in chunk:
                    scores[idx] = scores.get(idx, 0) + 0.1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[i] for i, _ in ranked]

# ----- RAG and Non-RAG Inference -----
def _model_device(m):
    # utility: find device of model parameters
    try:
        return next(m.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def generate_with_rag(prompt, max_tokens=200):
    retrieved = hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA)
    print("[DEBUG] Hybrid retrieved chunks:")
    for c in retrieved:
        print("-", c[:80].replace('\n', ' '), "...")
    context = "\n".join(retrieved)
    rag_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    device_for_model = _model_device(model)
    idx = torch.tensor([encode(rag_prompt[-256:])], dtype=torch.long).to(device_for_model)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens)
    return decode(out[0].tolist())

def generate_without_rag(prompt, max_tokens=200):
    device_for_model = _model_device(model)
    idx = torch.tensor([encode(prompt[-256:])], dtype=torch.long).to(device_for_model)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens)
    return decode(out[0].tolist())

# ----- Unified Generation Function -----
def generate(prompt, max_tokens, use_rag):
    # summarization route uses SmolLM (HF)
    if prompt.strip().lower().startswith("please summarize:"):
        print("[DEBUG] Using SmolLM summarizer")
        inputs = smol_tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            output_ids = smol_model.generate(**inputs, max_new_tokens=max_tokens)
        return smol_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if use_rag:
        print("[DEBUG] RAG mode enabled (hybrid retrieval)")
        return generate_with_rag(prompt, max_tokens)
    else:
        print("[DEBUG] RAG mode disabled (pure prompt)")
        return generate_without_rag(prompt, max_tokens)

# ----- Launch Gradio UI -----
gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Your Prompt", placeholder="Ask a question or give a prompt"),
        gr.Slider(50, 1000, step=50, value=200, label="Max Tokens"),
        gr.Checkbox(label="Use Retrieval-Augmented Generation (RAG)", value=True)
    ],
    outputs="text",
    title="Gītā-GPT (quantized-ready)",
    description="Toggle RAG on/off and use hybrid retrieval (embeddings + TF-IDF) with rare term boosting."
).launch()
