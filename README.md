# PDF-Extraction-using-PyMuPDF4LLM, LlamaIndex, OpenAI, Qdrant
## 🧠 Multi-Modal PDF Extraction and Retrieval App  
### Powered by `pymupdf4llm`, `LlamaIndex`, `CLIP Embeddings`, and `Qdrant`


### 📌 Overview

This Streamlit application enables **semantic search over both text and images extracted from PDF files**. It combines **advanced PDF parsing** with **multimodal embedding and vector search** to allow users to upload a PDF, extract its textual and visual content, and then query it using natural language.

The app uses:
- `pymupdf4llm` for layout-preserving PDF parsing and image extraction
- `LlamaIndex` for document processing and multi-modal vector indexing
- `CLIP Embeddings` for both text and image embeddings
- `Qdrant` as a high-performance vector store
- `Streamlit` for UI and interaction

---

### ⚙️ Features

✅ **Text + Image Extraction**  
- Uses `pymupdf4llm.to_markdown()` to extract text chunks and save images from each page  
- Converts raw data into `LlamaIndex Document` and `ImageNode` objects with rich metadata

✅ **Multimodal Vector Search**  
- Embeds text and images using `CLIP` model  
- Stores them separately in two Qdrant collections  
- Enables retrieval from both sources based on semantic similarity

✅ **Interactive Retrieval UI**  
- Search and retrieve **top-k most relevant text chunks and images** for any user query  
- Displays results in an intuitive layout (expandable text cards and image grid)

✅ **Real-Time Visualization**  
- Renders matched images in a clean subplot grid using `matplotlib`  
- Displays both image and score for precise understanding

---

### 🛠 Tech Stack

| Component      | Tech Used                                      |
|----------------|------------------------------------------------|
| 🧠 LLM/RAG      | `LlamaIndex` + `CLIPEmbedding`                 |
| 📄 PDF Parsing | `pymupdf4llm` (on top of PyMuPDF)               |
| 🧠 Embeddings   | `CLIP` (for text + image)                      |
| 🗂 Vector DB    | `Qdrant` (in-memory for this demo)             |
| 📊 Plotting     | `matplotlib`, `PIL`, `numpy`                   |
| 🖥 UI           | `Streamlit`                                    |

---

### 🔄 How It Works

1. **Upload a PDF**  
   Drag and drop a PDF via Streamlit sidebar.

2. **PDF Content Extraction**  
   - Uses `pymupdf4llm.to_markdown()` to extract structured text (per page) and images (saved as JPG)
   - Wraps text and metadata into `LlamaIndex.Document` objects
   - Loads images as `ImageNode` documents via `SimpleDirectoryReader`

3. **Embedding + Indexing**  
   - Uses `ClipEmbedding` to embed both text and image documents
   - Stores them in two separate Qdrant vector collections (`text_collection` & `image_collection`)

4. **Query and Retrieve**  
   - Users enter a natural language query
   - The system searches both text and image stores for semantically similar content
   - Results are scored and displayed visually with retrieval confidence

---

### 📁 Folder Structure

```
📦 pdf_multimodal_rag/
├── app.py                  # Main Streamlit app
├── images/                 # Extracted images from PDFs
├── requirements.txt        # Project dependencies
├── README.md               # GitHub project description
```


### 📸 Sample Use Case

> Upload a PDF contract with embedded diagrams or charts, and ask:  
> **“Show me the images related to payment section”**  
> → The system will retrieve relevant text from payment sections + matching visuals (e.g. invoice tables, chart images).

---

### 🧠 Project Highlights

- ✅ Layout-aware PDF parsing using `pymupdf4llm`
- ✅ Embeds visual and textual content with the same model for unified retrieval
- ✅ Uses `LlamaIndex`'s multimodal vector store with `Qdrant` as backend
- ✅ Displays images in a dynamic subplot grid with individual scores
- ✅ Fully extensible for custom embedding models or persistent Qdrant setup

---

### 🚀 Future Enhancements

- [ ] Enable persistent Qdrant storage (hosted or file-based)
- [ ] Add OCR fallback for scanned PDFs
- [ ] Support multi-turn conversations with document context
- [ ] Enable export of extracted Markdown and images
- [ ] Deploy on Hugging Face or Streamlit Cloud

---

### 🙌 Credits

- [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Qdrant](https://qdrant.tech/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
