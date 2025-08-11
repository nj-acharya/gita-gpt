import numpy as np
import faiss
import torch
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Constants ---
TOP_K = 5
ALPHA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "bg.txt"
CHUNK_SIZE = 500

# --- Load and chunk corpus ---
def load_and_chunk(file_path, chunk_size=CHUNK_SIZE):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = load_and_chunk(DATA_FILE)

# --- Embeddings and FAISS index ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# --- TF-IDF vectorizer ---
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)

# --- SmolLM summarizer ---
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct").to(DEVICE)

def summarize_with_smol(prompt, context):
    combined_input = f"{prompt.strip()} {context.strip()}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=300,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Hybrid retrieval ---
def hybrid_retrieve(prompt, top_k=TOP_K, alpha=ALPHA, rare_term_boost=True):
    query_embedding = embedder.encode([prompt])
    D_embed, I_embed = index.search(np.array(query_embedding), top_k * 2)

    tfidf_query = tfidf_vectorizer.transform([prompt])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]
    I_tfidf = np.argsort(tfidf_scores)[::-1][:top_k * 2]

    scores = {}
    for i in I_embed[0]:
        scores[i] = scores.get(i, 0) + alpha * (1 - D_embed[0][list(I_embed[0]).index(i)])
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

# --- Gradio interface function ---
def answer_question(prompt):
    context_chunks = hybrid_retrieve(prompt)
    context_text = "\n".join(context_chunks)

    if prompt.lower().startswith("please summarize:"):
        summary = summarize_with_smol(prompt, context_text)
        return summary.strip()
    else:
        return "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

# --- Gradio app ---
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Enter your question or instruction", placeholder="Please summarize: What is dharma?", lines=2),
    outputs=gr.Textbox(label="Response"),
    title="Gita-GPT Hybrid RAG with SmolLM",
    description="Type a question. If you start your prompt with 'Please summarize:', it will use SmolLM-135M-Instruct."
)

if __name__ == "__main__":
    demo.launch()
