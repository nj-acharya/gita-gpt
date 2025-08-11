# hybrid_rag_smol_gpu_benchmark.py
# Converted from user's uploaded script. See original: :contentReference[oaicite:1]{index=1}

import time
import os
import numpy as np
import faiss
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import statistics
import argparse
import subprocess
import json

# -------------------------
# Config / CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data-file", default="bg.txt")
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--chunk-size", type=int, default=500)
parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
parser.add_argument("--load-in-8bit", action="store_true", help="Use bitsandbytes 8-bit load (recommended for low-memory GPUs)")
parser.add_argument("--benchmark-queries", default=None, help="Path to JSON list of queries for automated benchmark")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()

TOP_K = args.top_k
ALPHA = args.alpha
CHUNK_SIZE = args.chunk_size
DATA_FILE = args.data_file

# -------------------------
# Helper: chunking data
# -------------------------
def load_and_chunk(file_path, chunk_size=500):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = load_and_chunk(DATA_FILE, CHUNK_SIZE)

# -------------------------
# Embedding/index setup
# -------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# -------------------------
# TF-IDF
# -------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)

# -------------------------
# Model loading (GPU-first with optional 8-bit)
# -------------------------
device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
print(f"Selected device: {device}")

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

# attempt to load in 8-bit on GPU if requested
model = None
use_8bit = args.load_in_8bit and device == "cuda"
try:
    if use_8bit:
        print("Attempting to load model in 8-bit (bitsandbytes)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16  # bitsandbytes keeps weights in 8-bit but model uses fp16 for ops
        )
    elif device == "cuda":
        print("Loading model in fp16 on CUDA...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("Loading model on CPU (fp32). This will be slow but acts as fallback.")
        model = AutoModelForCausalLM.from_pretrained(args.model)
except Exception as e:
    print("Model load failed:", e)
    print("Falling back to CPU (fp32) load.")
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model)

model.eval()

# -------------------------
# Summarization function (uses no_grad + generation config)
# -------------------------
def summarize_with_smol(prompt, context, max_new_tokens=300):
    combined_input = f"{prompt.strip()} {context.strip()}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Hybrid retrieval (unchanged but cleaned)
# -------------------------
def hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA, rare_term_boost=True):
    query_embedding = embedder.encode([prompt])
    D_embed, I_embed = index.search(np.array(query_embedding), top_k * 2)
    tfidf_query = tfidf_vectorizer.transform([prompt])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]
    I_tfidf = np.argsort(tfidf_scores)[::-1][:top_k * 2]

    scores = {}
    # embeddings contribute: use (1 - distance) as score
    for i_idx, i in enumerate(I_embed[0]):
        scores[i] = scores.get(i, 0) + alpha * (1 - D_embed[0][i_idx])
    for i in I_tfidf:
        scores[i] = scores.get(i, 0) + (1 - alpha) * tfidf_scores[i]

    if rare_term_boost:
        tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
        tfidf_query_vector = tfidf_query.toarray()[0]
        top_term_indices = np.argsort(tfidf_query_vector)[::-1]
        rare_terms = [tfidf_vocab[i] for i in top_term_indices[:3] if tfidf_query_vector[i] > 0]
        for idx, chunk in enumerate(chunks):
            if any(term in chunk for term in rare_terms):
                scores[idx] = scores.get(idx, 0) + 0.1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[i] for i, _ in ranked]

# -------------------------
# GPU memory / nvidia-smi helper
# -------------------------
def get_gpu_mem_info():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,nounits,noheader"], encoding="utf-8")
        total, used = map(int, out.strip().split(","))
        return {"total_mb": total, "used_mb": used}
    except Exception:
        return None

# -------------------------
# Benchmark harness
# -------------------------
def benchmark_queries(queries, warmup=2, runs=10):
    # warmup
    for i in range(warmup):
        q = queries[i % len(queries)]
        ctx = "\n".join(hybrid_retrieve(q))
        _ = summarize_with_smol(q, ctx, max_new_tokens=64)

    latencies = []
    for i in range(runs):
        q = queries[i % len(queries)]
        start = time.time()
        ctx = "\n".join(hybrid_retrieve(q))
        t_retrieval = time.time()
        _ = summarize_with_smol(q, ctx, max_new_tokens=64)
        end = time.time()
        latencies.append(end - start)
        # optional: record GPU memory
        gpu_info = get_gpu_mem_info()
        print(f"run {i+1}/{runs} latency={latencies[-1]:.3f}s retrieval_time={t_retrieval-start:.3f}s gpu_info={gpu_info}")
    return {
        "runs": runs,
        "latencies": latencies,
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=100)[94],
        "mean": statistics.mean(latencies)
    }

# -------------------------
# If invoked with --benchmark-queries, run automated benchmark
# -------------------------
if args.benchmark_queries:
    with open(args.benchmark_queries, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print("Running benchmark on", len(queries), "queries")
    results = benchmark_queries(queries, warmup=args.warmup, runs=args.runs)
    print("Benchmark summary:", results)
    exit(0)

# -------------------------
# Interactive loop (same UX as original)
# -------------------------
if __name__ == "__main__":
    print("Ready. Type a question, or 'exit'. For automated benchmarking pass --benchmark-queries queries.json.")
    while True:
        prompt = input("\nAsk a question (or type 'exit'): ").strip()
        if prompt.lower() == "exit":
            break

        context_chunks = hybrid_retrieve(prompt)
        context_text = "\n".join(context_chunks)

        if prompt.lower().startswith("please summarize:"):
            print("\n[Summary]")
            t0 = time.time()
            summary = summarize_with_smol(prompt, context_text)
            t1 = time.time()
            print(summary.strip())
            print(f"\n(inference time: {t1-t0:.3f}s)")
        else:
            print("\n[Top Retrieved Chunks]\n")
            for i, chunk in enumerate(context_chunks, 1):
                print(f"--- Chunk {i} ---\n{chunk.strip()}\n")
