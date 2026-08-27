# 🧴 Skincare RAG Chatbot

An **AI-powered skincare chatbot** built using **Retrieval-Augmented Generation (RAG)**. The chatbot retrieves relevant skincare information from a knowledge base and uses a Large Language Model (LLM) to generate contextual and helpful responses.

The project combines **LangChain, ChromaDB, Hugging Face embeddings, Sentence Transformers, and Groq LLMs** with a **Streamlit** interface.

---

## 🚀 Features

* 🔍 **Retrieval-Augmented Generation (RAG)**
* 🧠 Semantic search using Hugging Face embeddings
* 📚 Document ingestion and text chunking
* 🗄️ Vector storage using **ChromaDB**
* 🤖 LLM-powered responses using **Groq**
* 💬 Interactive chatbot interface using **Streamlit**
* ⚡ Fast response generation
* 📄 Supports knowledge bases created from skincare documents/PDFs

---

## 🏗️ Project Architecture

```text
                 ┌─────────────────────┐
                 │   Skincare Documents │
                 │      / PDFs          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Document Loader    │
                 │    & Processing     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Text Splitter    │
                 │  Chunk Documents    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Hugging Face        │
                 │ Embeddings          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     ChromaDB        │
                 │   Vector Database   │
                 └──────────┬──────────┘
                            │
                    User Question
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Similarity Search   │
                 │   / Retrieval       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Groq LLM        │
                 │  Response Generation│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Streamlit Chat UI  │
                 └─────────────────────┘
```

---

## 📂 Project Structure

```text
Skincare_RAG/
│
├── 📄 app.py                 # Streamlit chatbot application
├── 📄 ingest.py              # Document ingestion and vector database creation
├── 📄 requirements.txt       # Python dependencies
├── 📄 .env                   # API keys and environment variables
├── 📄 .gitignore             # Files excluded from Git
│
├── 📁 data/
│   └── skincare_documents/   # Skincare knowledge base
│
├── 📁 chroma_db/             # ChromaDB vector database
│
└── 📄 README.md              # Project documentation
```

> **Note:** Your actual filenames/folder structure may be different. Update this section according to your repository.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Skincare_RAG.git
```

Navigate into the project:

```bash
cd Skincare_RAG
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt` yet, you can generate it with:

```bash
pip freeze > requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace:

```text
your_groq_api_key
```
---

# 📚 Building the Knowledge Base

Before running the chatbot, ingest your skincare documents into the vector database.

Run:

```bash
python ingest.py
```

The ingestion pipeline performs the following steps:

```text
Documents
    ↓
Document Loading
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB
```

The resulting embeddings are stored in the Chroma vector database and can later be searched by the chatbot.

---

# 💬 Running the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser and start asking skincare-related questions.

---

# 🧠 How RAG Works

This project uses **Retrieval-Augmented Generation** instead of relying only on the LLM's pre-trained knowledge.

For example, when a user asks:

```text
What are the common causes of acne?
```

The system performs:

### 1. Query

The user enters a skincare question.

### 2. Embedding

The question is converted into a numerical vector using a Hugging Face/Sentence Transformers embedding model.

### 3. Retrieval

ChromaDB searches the knowledge base for documents that are semantically similar to the question.

### 4. Context

The most relevant document chunks are retrieved.

### 5. Generation

The retrieved context is passed to the Groq-powered LLM.

### 6. Response

The LLM generates an answer based on the retrieved skincare information.

```text
User Question
      ↓
Embedding Model
      ↓
Vector Search
      ↓
ChromaDB
      ↓
Relevant Documents
      ↓
Groq LLM
      ↓
Generated Answer
```

---

# 🧪 Example Questions

You can ask questions such as:

```text
What are the common causes of acne?

What skincare ingredients are useful for oily skin?

How can I build a basic skincare routine?

What is the difference between dry skin and dehydrated skin?

What ingredients should I look for in a moisturizer?

How can I take care of sensitive skin?

What are common causes of skin irritation?
```

---

# 📦 Requirements

The project uses the following major Python packages:

```text
pandas
streamlit
python-dotenv
langchain-community
langchain-text-splitters
langchain-huggingface
torch
transformers
sentence-transformers
langchain-groq
chromadb
```

You can install them using:

```bash
pip install -r requirements.txt
```

---

# ⚠️ Disclaimer

This chatbot is intended for **educational and informational purposes only**.

The information provided by the chatbot should not be considered professional medical advice, diagnosis, or treatment.

For persistent, severe, or concerning skin conditions, users should consult a qualified dermatologist or healthcare professional.

---


# 👨‍💻 Author

**Thamo Tharan**

Built as an AI/RAG project to explore **Retrieval-Augmented Generation, vector databases, embeddings, and LLM applications**.

---

## ⭐ If You Find This Project Useful

Give the repository a ⭐ on GitHub!

Feel free to fork the project and experiment with the RAG pipeline.
