# 🧾 LexiParse – Legal Document Analyzer

LexiParse is a simple tool that extracts insights from legal documents using LLMs via Ollama. It supports plain text and structured inputs, offering summarization and question-answering features.

## ✨ Features
- Summarize legal contracts or clauses
- Ask questions about uploaded documents
- Uses local LLMs via [Ollama](https://ollama.com/)

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Sample Usage
Try it out with the included 'sample_document.txt'.

### 🧠 Tech Stack
Python

Ollama LLMs

Streamlit (if used)

LangChain (optional, depending on usage)

### 📂 Project Structure
```bash
app.py              # Main application logic
ollama_utils.py     # LLM query functions
requirements.txt    # Python dependencies
sample_document.txt # Example legal document
```
