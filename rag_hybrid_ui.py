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

# ----- Configuration -----
CORPUS_DIR = "docs/"
CHUNK_SIZE = 300
STRIDE = 150  # 50% overlap
TOP_K = 3
ALPHA = 0.5  # hybrid weighting: 1 = pure embedding, 0 = pure tf-idf
EMBED_MODEL = 'all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Load and Chunk Documents with Sliding Window -----
print("Loading and chunking documents with sliding window...")

def sliding_window_chunk(text, chunk_size=300, stride=150):
    tokens = textwrap.wrap(text, width=1, break_long_words=False)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = ''.join(tokens[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

documents = []
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(CORPUS_DIR, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())

chunks = []
for doc in documents:
    chunks.extend(sliding_window_chunk(doc, chunk_size=CHUNK_SIZE, stride=STRIDE))

print(f"Loaded {len(documents)} documents with {len(chunks)} total chunks.")

# ----- Embed Chunks -----
print("Embedding chunks...")
embedder = SentenceTransformer(EMBED_MODEL)
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# ----- Build FAISS Index -----
print("Building FAISS index...")
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# ----- Build TF-IDF Index -----
print("Building TF-IDF index...")
tfidf_vectorizer = TfidfVectorizer().fit(chunks)
tfidf_matrix = tfidf_vectorizer.transform(chunks)

# ----- Load GPT Model -----
print("Loading GPT model...")
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt.pth', map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ----- Load Vocab -----
with open('vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ----- Hybrid Retrieval Function -----
def hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA):
    query_embedding = embedder.encode([prompt])
    D_embed, I_embed = index.search(np.array(query_embedding), top_k)

    tfidf_query = tfidf_vectorizer.transform([prompt])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]
    I_tfidf = np.argsort(tfidf_scores)[::-1][:top_k]

    scores = {}
    for i in I_embed[0]:
        scores[i] = scores.get(i, 0) + alpha * (1 - D_embed[0][list(I_embed[0]).index(i)])
    for i in I_tfidf:
        scores[i] = scores.get(i, 0) + (1 - alpha) * tfidf_scores[i]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[i] for i, _ in ranked]

# ----- RAG and Non-RAG Inference -----
def generate_with_rag(prompt, max_tokens=200):
    retrieved = hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA)
    print("[DEBUG] Hybrid retrieved chunks:")
    for c in retrieved:
        print("-", c[:80].replace('\n', ' '), "...")

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
    title="Gītā-GPT with Hybrid RAG and Sliding Window Chunking",
    description="Toggle RAG on/off and use hybrid retrieval (embeddings + TF-IDF) with sliding window chunking."
).launch()
