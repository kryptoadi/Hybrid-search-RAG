# 🔎 Hybrid Search RAG

**A Retrieval-Augmented Generation application that combines keyword-based and semantic search to answer questions from user-uploaded PDF documents.**

Hybrid Search RAG uses BM25, FAISS, and Reciprocal Rank Fusion (RRF) to retrieve relevant document chunks before passing them to a Groq-powered LLM for answer generation.

Users can upload their own PDF files, ask questions about their contents, and inspect the retrieved text used to generate answers.

## ✨ Features

- 📄 **Dynamic PDF Upload:** Upload one or multiple PDF documents through the Streamlit interface.
- ✂️ **Text Chunking:** Split extracted document text into smaller, overlapping chunks using LangChain.
- 🔤 **Keyword Search:** Use BM25Okapi to retrieve chunks based on exact word matches.
- 🧠 **Semantic Search:** Generate embeddings and retrieve semantically similar chunks using FAISS.
- 🔀 **Hybrid Retrieval:** Combine BM25 and FAISS rankings using Reciprocal Rank Fusion.
- 🤖 **LLM-Powered Answers:** Generate context-grounded answers using Groq and LangChain.
- 📝 **Prompt Engineering:** Use LangChain PromptTemplate to structure the context and question passed to the LLM.
- 📚 **Source References:** Display source filenames and page numbers associated with retrieved chunks.
- 🔍 **Retrieval Transparency:** Inspect retrieved text to understand the evidence behind generated answers.
- 🖥️ **Interactive Interface:** Upload documents and ask questions through Streamlit.

## 🏗️ Architecture

```text
                User Uploads PDF Files
                         |
                         v
                 PDF Text Extraction
                         |
                         v
                   Text Chunking
                         |
                         v
              +----------+----------+
              |                     |
              v                     v
        BM25Okapi Search       FAISS Search
        (Keyword Search)      (Semantic Search)
              |                     |
              v                     v
              +----------+----------+
                         |
                         v
              Reciprocal Rank Fusion
                         |
                         v
                Top Relevant Chunks
                         |
                         v
                  PromptTemplate
                         |
                         v
                    Groq LLM
                         |
                         v
                 Generated Answer
                         |
                         v
              Answer + Source References
```

## 🛠️ Tech Stack

| Technology                      | Purpose                                 |
| ------------------------------- | --------------------------------------- |
| Python                          | Core programming language               |
| Streamlit                       | Interactive web interface               |
| LangChain                       | Document processing and LLM integration |
| PyPDFLoader                     | PDF text extraction                     |
| RecursiveCharacterTextSplitter  | Text chunking                           |
| Google Generative AI Embeddings | Text embedding generation               |
| FAISS                           | Dense vector similarity search          |
| rank_bm25                       | BM25 keyword retrieval                  |
| Reciprocal Rank Fusion          | Combining ranked search results         |
| Groq                            | LLM inference                           |
| PromptTemplate                  | Structured prompt construction          |
| python-dotenv                   | Environment variable management         |

## 🔍 How Hybrid Search Works

Traditional RAG applications often rely on semantic vector search alone. However, embedding-based retrieval can sometimes miss exact technical terms, identifiers, or specific keywords.

This project combines two complementary retrieval methods.

### 1. BM25 Keyword Search

BM25Okapi ranks document chunks based on lexical relevance, considering factors such as term frequency, word importance, and document length.

It is particularly useful for queries containing exact terms, product names, technical terminology, and numerical identifiers.

### 2. FAISS Semantic Search

FAISS retrieves document chunks based on vector similarity. Embeddings allow the system to find conceptually related content even when the wording differs from the query.

### 3. Reciprocal Rank Fusion (RRF)

RRF combines the rankings produced by BM25 and FAISS.

The score for a document is calculated as:

[
\operatorname{RRF}(d)=
\sum\_{r \in R}\frac{1}{k+\operatorname{rank}\_r(d)}
]

Where:

- (d) is a document chunk.
- (R) is the set of retrieval result lists.
- (\operatorname{rank}\_r(d)) is the chunk's position in a result list.
- (k) is a ranking constant, commonly set to 60.

RRF uses the ranks rather than directly combining raw BM25 and vector similarity scores.

The resulting ranked chunks are passed to the LLM as context for answer generation.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or newer
- Git
- A Groq API key
- A Google AI API key for embedding generation

### 1. Clone the repository

```bash
git clone https://github.com/kryptoadi/Hybrid-search-RAG.git
cd hybrid-search-rag
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Replace the placeholders with your own API keys.

**Security:** Never commit `.env` or expose API keys in source code, screenshots, or public repositories.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, upload your PDF documents, and start asking questions.

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload one or more PDF documents.
3. Click **Process PDFs** to extract, split, embed, and index their contents.
4. Enter a question about the uploaded documents.
5. Click **Get Answer**.
6. Read the generated response.
7. Expand the source sections to inspect the retrieved text and associated PDF pages.

## 📂 Project Structure

```text
hybrid-search-rag/
│
├── app.py                 # Streamlit interface and document indexing
├── hybrid_retriever.py    # BM25, FAISS fusion, RRF, and Groq answers
├── requirements.txt       # Python dependencies
├── .env                   # Local API keys (not committed)
├── .gitignore             # Files excluded from Git
└── README.md              # Project documentation
```

## 🎯 Example Use Cases

- Querying technical documentation
- Searching company policies and process manuals
- Extracting information from product documentation
- Answering questions about reports and research papers
- Exploring knowledge bases using natural-language questions

## ⚠️ Current Limitations

- Retrieval quality depends on document quality, chunking, embeddings, and ranking.
- Scanned PDFs may require OCR before their text can be searched.
- The current BM25 tokenizer uses basic whitespace tokenization.
- Generated answers can still contain inaccuracies; retrieved passages should be checked.
- Source references identify retrieved chunks and do not guarantee that every answer statement is fully supported.
- The current implementation builds indexes when documents are processed; persistent storage and production-scale multi-user handling are not yet implemented.
- API access, model availability, and usage limits depend on the configured providers.

## 🔮 Future Improvements

- [ ] Evaluate BM25-only, FAISS-only, and hybrid retrieval on a fixed question set.
- [ ] Add Precision\@K, Recall\@K, Mean Reciprocal Rank (MRR), and other retrieval metrics.
- [ ] Experiment with different chunk sizes and overlap values.
- [ ] Add a cross-encoder reranker.
- [ ] Improve source citation display and retrieved-chunk inspection.
- [ ] Add conversational memory for follow-up questions.
- [ ] Improve query tokenization and preprocessing.
- [ ] Add automated tests and error handling.
- [ ] Support persistent indexes and more robust multi-user sessions.
- [ ] Deploy the application.

## 👨‍💻 Author

**Aditya Kumar**

Computer Science and Engineering | AI, Machine Learning, and Generative AI

This project explores hybrid information retrieval and Retrieval-Augmented Generation using open-source retrieval tools and LLM APIs.

---

⭐ If you find this project useful, consider starring the repository.

**Note:** This project is intended for educational and demonstration purposes. Verify generated answers against the original source documents before relying on them for important decisions.
