## CHANGELOG for `gradio_ui.py`

### [v1.1.0] - Added RAG, Sliding Window, Debug Support

#### ✅ RAG Toggle Support
- Added a Gradio checkbox to enable/disable Retrieval-Augmented Generation (RAG).
- Users can choose to use document context or rely solely on the prompt.

#### 🧠 Retrieval-Augmented Generation (RAG)
- Uses SentenceTransformer to generate embeddings for document chunks.
- Constructs a FAISS index for fast approximate nearest neighbor search.
- Prepends the top-k similar chunks as context to the input prompt before generation.

#### 🔁 Sliding Window Chunking
- Introduced sliding window strategy to chunk documents:
  - `CHUNK_SIZE = 300`
  - `STRIDE = 150`
- Improves information retention across chunk boundaries.

#### 🐛 Debug Logging
- Added `[DEBUG]` print statements to indicate RAG mode and document stats:
  - RAG enabled/disabled.
  - Number of documents loaded.
  - Number of chunks generated with sliding window.

#### 🔤 Prompt Truncation and Vocab Handling
- Keeps only the last 256 characters of prompt/context when encoding.
- Uses `vocab.pkl` to encode/decode input/output sequences.

#### 🎛️ Gradio Interface Enhancements
- Inputs:
  - `Textbox` for user prompt.
  - `Slider` for max tokens.
  - `Checkbox` to toggle RAG.
- Output:
  - Textbox for generated response.
- Updated title and description for clarity.

---

*File last modified with enhancements on 2025-08-01.*

