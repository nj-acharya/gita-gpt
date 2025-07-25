import os
import torch
import faiss
import pickle
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import textwrap
from gpt import GPTLanguageModel

# ----- Configuration -----
CORPUS_DIR = "docs/"  # directory containing .txt files
CHUNK_SIZE = 300       # characters per chunk
TOP_K = 3              # number of chunks to retrieve
EMBED_MODEL = 'all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Load and Chunk Corpus -----
print("Loading corpus from", CORPUS_DIR)
documents = []
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(CORPUS_DIR, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())

print(f"Loaded {len(documents)} documents. Chunking...")
chunks = []
for doc in documents:
    chunks.extend(textwrap.wrap(doc, width=CHUNK_SIZE, break_long_words=False))

# ----- Embed Chunks -----
print("Embedding", len(chunks), "chunks...")
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

# Load vocab
with open('vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ----- Inference Function with RAG -----
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

# ----- Gradio Web UI -----
gr.Interface(
    fn=generate_with_rag,
    inputs=[
        gr.Textbox(label="Your Question", placeholder="Ask something based on the document corpus"),
        gr.Slider(50, 1000, step=50, value=200, label="Max Tokens")
    ],
    outputs="text",
    title="Gītā-GPT with Retrieval-Augmented Generation",
    description="Ask a question grounded in your uploaded document corpus."
).launch()
