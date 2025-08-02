import os
import torch
import faiss
import pickle
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from gpt import GPTLanguageModel

# ----- Configuration -----
CORPUS_DIR = "docs/"        # Directory containing .txt files
CHUNK_SIZE = 300            # Characters per chunk
STRIDE = 150                # Sliding window stride (50% overlap)
TOP_K = 3                   # Number of chunks retrieved
EMBED_MODEL = 'all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Load and Chunk Corpus -----
def sliding_window_chunks(text, size=300, stride=150):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += stride
    return chunks

print("Loading and chunking documents...")
documents = []
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(CORPUS_DIR, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())

chunks = []
for doc in documents:
    chunks.extend(sliding_window_chunks(doc, size=CHUNK_SIZE, stride=STRIDE))

print(f"[DEBUG] Loaded {len(documents)} documents with {len(chunks)} sliding window chunks.")

# ----- Embed Chunks -----
print("Embedding chunks...")
embedder = SentenceTransformer(EMBED_MODEL)
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# ----- Build FAISS Index -----
print("Building FAISS index...")
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# ----- Load GPT Model -----
print("Loading GPT model...")
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- Load Vocabulary -----
with open('vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ----- RAG and Non-RAG Inference Functions -----
def generate_with_rag(prompt, max_tokens=200):
    query_embedding = embedder.encode([prompt])
    D, I = index.search(np.array(query_embedding), TOP_K)
    retrieved = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved)
    rag_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    idx = torch.tensor([encode(rag_prompt[-256:])], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens)
    return decode(out[0].tolist())

def generate_without_rag(prompt, max_tokens=200):
    idx = torch.tensor([encode(prompt[-256:])], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens)
    return decode(out[0].tolist())

# ----- Unified Generation Function -----
def generate(prompt, max_tokens, use_rag):
    if use_rag:
        print("[DEBUG] RAG mode enabled. Retrieving context...")
        return generate_with_rag(prompt, max_tokens)
    else:
        print("[DEBUG] RAG mode disabled. Using prompt only.")
        return generate_without_rag(prompt, max_tokens)

# ----- Gradio Interface -----
gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Your Prompt", placeholder="Ask a question or give a prompt"),
        gr.Slider(50, 1000, step=50, value=200, label="Max Tokens"),
        gr.Checkbox(label="Use Retrieval-Augmented Generation (RAG)", value=True)
    ],
    outputs="text",
    title="Gītā-GPT with Optional RAG & Sliding Window",
    description="Toggle RAG on/off. Now uses sliding window chunking for better context coverage."
).launch()
