import os
import io
import re
import uuid
import tempfile
import requests
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, List  # Fix typing imports
import pandas as pd
from dotenv import load_dotenv
import sqlite3

# Handle optional dependencies
PDF_AVAILABLE = False
DOCX_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    
    PDF_AVAILABLE = True
except ImportError:
    st.error("PyPDF2 not available. PDF processing will be disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    st.warning("python-docx not available. DOCX processing will be disabled.")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.warning("transformers not available. NER features will be disabled.")

# Import OpenAI utilities
from ollama_utils import ask_ollama_system_prompt  # Replace groq_utils import

# Load environment variables
load_dotenv()

# Database setup
def init_db():
    conn = sqlite3.connect('legal_docs.db')
    c = conn.cursor()
    c.execute('''

    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        name TEXT,
        content TEXT,
        summary TEXT,
        upload_date TEXT,
        doc_type TEXT
    )
    ''')
    conn.commit()
    return conn

# Initialize NER pipeline if available
@st.cache_resource
def load_ner_model():
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
            model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
            return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception as e:
            st.warning(f"Could not load NER model: {str(e)}")
            return None
    return None

# Initialize NER model if available
ner_model = load_ner_model() if TRANSFORMERS_AVAILABLE else None

# Document processing functions
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from a docx file."""
    if not DOCX_AVAILABLE:
        st.error("The python-docx library is not available. DOCX files cannot be processed.")
        return ""
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(docx_file.getvalue())
            tmp_path = tmp.name
        
        # Extract text
        doc = Document(tmp_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        # Remove temp file
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_entities(text):
    """Extract named entities from text using NER."""
    if not ner_model:
        st.warning("Named Entity Recognition is not available.")
        return []
        
    try:
        # Limit text length to avoid timeout
        max_length = 10000
        entities = ner_model(text[:max_length])
        return entities
    except Exception as e:
        st.error(f"Error in NER extraction: {str(e)}")
        return []

def display_entities(entities):
    """Format and display entities in a more organized way."""
    if not entities:
        st.write("No significant entities detected.")
        return
        
    # Define important entity types for legal documents
    legal_entity_groups = {
        'ORG': 'Organizations',
        'LOC': 'Locations',
        'PER': 'Persons',
        'DATE': 'Dates',
        'MONEY': 'Monetary Values'
    }
    
    # Filter and organize entities
    entity_groups = {}
    for entity in entities:
        group = entity['entity_group']
        word = entity['word'].strip()
        
        # Skip unwanted entities
        if len(word) <= 1 or word in ['##L', 'C', 'CR']:
            continue
        if group not in legal_entity_groups:
            continue
            
        if group not in entity_groups:
            entity_groups[group] = set()
        entity_groups[group].add(word)

    # Display entities by group with better formatting
    for group, words in entity_groups.items():
        if words:
            st.write(f"**{legal_entity_groups.get(group, group)}:**")
            st.write(", ".join(sorted(words)))

def detect_legal_citations(text):
    """Detect common legal citations in the document."""
    citation_patterns = [
        # Case citations
        r'\b\d{1,3}\s+U\.S\.\s+\d{1,4}\b',  # US Reports
        r'\b\d{1,3}\s+S\.\s*Ct\.\s+\d{1,4}\b',  # Supreme Court Reporter
        r'\b\d{1,3}\s+F\.\s*\d{1,3}d\s+\d{1,4}\b',  # Federal Reporter 3d
        r'\b\d{1,3}\s+F\.\s*Supp\.\s*\d{1,3}d\s+\d{1,4}\b',  # Federal Supplement
        
        # Statute citations
        r'\b\d{1,2}\s+U\.S\.C\.\s+§\s*\d{1,5}\b',  # US Code
        r'\b\d{1,2}\s+C\.F\.R\.\s+§\s*\d{1,5}\b',  # Code of Federal Regulations
    ]
    
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
        
    return citations

def calculate_document_metadata(text):
    """Calculate basic document metadata"""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    return {
        "Word count": word_count,
        "Character count": char_count,
        "Sentence count": sentence_count,
        "Estimated reading time": f"{round(word_count / 250, 1)} minutes"
    }

def save_document(name, content, summary, doc_type):
    """Save document to database"""
    conn = init_db()
    c = conn.cursor()
    doc_id = str(uuid.uuid4())
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute(
        "INSERT INTO documents (id, name, content, summary, upload_date, doc_type) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, name, content, summary, upload_date, doc_type)
    )
    conn.commit()
    conn.close()
    return doc_id

def load_document(doc_id):
    """Load document from database by ID"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return None
    
    return {
        "id": result[0],
        "name": result[1],
        "content": result[2],
        "summary": result[3],
        "upload_date": result[4],
        "doc_type": result[5]
    }

def get_all_documents():
    """Get all documents from database"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, name, upload_date, doc_type FROM documents ORDER BY upload_date DESC")
    results = c.fetchall()
    conn.close()
    
    docs = []
    for result in results:
        docs.append({
            "id": result[0],
            "name": result[1],
            "upload_date": result[2],
            "doc_type": result[3]
        })
    
    return docs

def get_document_answer(document_text, question):
    """Get answer to a question about the document using Ollama"""
    task_prompt = f"""You are a legal expert assistant. Answer the following question about the legal document in a clear, concise manner. If the document doesn't contain the information needed to answer the question, say so clearly.

Question: {question}"""
    
    answer = ask_ollama_system_prompt(document_text, task_prompt)
    return answer

def compare_documents(doc1_text, doc2_text):
    """Compare two legal documents using Ollama"""
    task_prompt = """You are a legal document analyzer specialized in document comparison. Compare these two documents and identify:
    1. Key similarities
    2. Important differences
    3. Changes in legal terms or conditions
    4. Variations in obligations or requirements
    
    Focus on substantive legal differences rather than formatting or minor wording changes."""
    
    prompt = f"""Document 1:
{doc1_text[:12000]}

Document 2:
{doc2_text[:12000]}"""
    
    comparison = ask_ollama_system_prompt(prompt, task_prompt)
    return comparison

def analyze_document(text: str) -> Dict[str, Any]:
    """Analyze document content using local Llama2 instance."""
    if not text:
        return {"error": "No text provided for analysis"}

    system_prompt = """Analyze this legal document and provide the following sections:

# 📄 Document Analysis

## 📋 Summary
Provide a concise overview of the agreement

## 👥 Key Parties
List the main parties and their roles

## 💰 Financial Terms
- Total amount
- Payment schedule
- Additional fees

## 📅 Important Dates
- Agreement date
- Duration
- Key deadlines

## ✍️ Key Obligations
List main responsibilities for each party

## ⚖️ Legal Terms
- Governing law
- Dispute resolution
- Termination conditions

Format the response using markdown with clear sections and emoji headers."""

    with st.spinner("Analyzing document with Llama2... ⏳"):
        result = ask_ollama_system_prompt(text, system_prompt)
        
        if result.startswith("Error:"):
            return {"error": result}
        
        return {
            "summary": result,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

# Main app
def main():
    # Initialize the database
    init_db()
    
    # Initialize session state if needed
    if 'show_demo' not in st.session_state:
        st.session_state.show_demo = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    
    # Sidebar navigation
    st.sidebar.title("📜 Legal Document Analyzer")
    page = st.sidebar.radio("Navigation", [
        "Home",
        "Document Analysis", 
        "Document Storage", 
        "Q&A With Documents",
        "Document Comparison"
    ])
    
    # Home page
    if page == "Home":
        st.title("Legal Document Analysis Assistant")
        st.write("""
        Welcome to the Legal Document Analyzer! This tool helps you analyze, store, and interact with legal documents.
        
        **Features:**
        - Extract key information from legal documents
        - Summarize legal documents automatically
        - Store documents for later reference
        - Ask questions about your documents
        - Compare multiple legal documents
        - Detect legal citations and entities
        
        Select a function from the sidebar to get started.
        """)
        
        # Display sample analysis if user wants to see a demo
        if st.button("Show Demo"):
            st.session_state.show_demo = True
            
        if st.session_state.get("show_demo", False):
            st.write("### Sample Analysis")
            st.write("Here's what a document analysis would look like:")
            st.info("Document Summary: This contract outlines the terms of a commercial lease agreement between ABC Properties LLC (Landlord) and XYZ Corporation (Tenant) for the premises located at 123 Business Avenue. The lease term is 5 years with an annual rent of $240,000 payable in monthly installments.")
            
            st.write("**Detected Legal Entities:**")
            st.write("**Organizations:** ABC Properties LLC, XYZ Corporation")
            st.write("**Locations:** 123 Business Avenue")
            st.write("**Dates:** 5 years")
            st.write("**Monetary Values:** $240,000")
            
            st.write("**Document Metadata:**")
            metadata_demo = {"Word count": 1254, "Character count": 7890, "Sentence count": 45, "Estimated reading time": "5.0 minutes"}
            st.json(metadata_demo)
    
    # Document Analysis page
    elif page == "Document Analysis":
        st.title("Document Analysis")
        
        # Initialize variables
        doc_type = "TXT"
        file_name = "Pasted Document"
        
        doc_option = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"]
        )
        
        if doc_option == "Upload File":
            uploaded_file = st.file_uploader("Upload legal document", 
                                           type=["pdf", "docx", "txt"] if DOCX_AVAILABLE else ["pdf", "txt"])
            
            if uploaded_file is not None:
                file_name = uploaded_file.name
                
                # Process based on file type
                if file_name.endswith('.pdf'):
                    st.session_state.document_text = extract_text_from_pdf(uploaded_file)
                    doc_type = "PDF"
                elif file_name.endswith('.docx') and DOCX_AVAILABLE:
                    st.session_state.document_text = extract_text_from_docx(uploaded_file)
                    doc_type = "DOCX"
                else:  # txt file
                    st.session_state.document_text = uploaded_file.getvalue().decode("utf-8")
                    doc_type = "TXT"
                
                if st.session_state.document_text:
                    st.success(f"File '{file_name}' uploaded successfully!")
                else:
                    st.error("Failed to extract text from the document.")
        else:
            st.session_state.document_text = st.text_area("Paste your legal document here:", height=300)
            
        # Only show analysis options if we have document text
        if st.session_state.document_text:
            # Preview of document
            with st.expander("Document Preview"):
                st.write(st.session_state.document_text[:1000] + "..." if len(st.session_state.document_text) > 1000 else st.session_state.document_text)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Analyze Document"):
                    st.session_state.analysis_result = analyze_document(st.session_state.document_text)
            
            # Display results if analysis was performed
            if st.session_state.analysis_result:
                if "error" in st.session_state.analysis_result:
                    st.error(st.session_state.analysis_result["error"])
                else:
                    st.subheader("📄 Document Analysis")
                    st.markdown(st.session_state.analysis_result["summary"])
                    
                    if ner_model and TRANSFORMERS_AVAILABLE:
                        st.subheader("🔎 Named Entities")
                        entities = extract_entities(st.session_state.document_text)
                        display_entities(entities)
                    
                    # Save button in second column
                    with col2:
                        if st.button("Save Document to Storage"):
                            summary = st.session_state.analysis_result.get("summary", "No summary available")
                            doc_id = save_document(file_name, st.session_state.document_text, summary, doc_type)
                            st.success(f"Document saved successfully! Document ID: {doc_id}")
    
    # Document Storage page
    elif page == "Document Storage":
        st.title("Document Storage")
        
        # Get all documents from database
        documents = get_all_documents()
        
        if not documents:
            st.info("No documents stored yet. Analyze and save documents to see them here.")
        else:
            st.write(f"You have {len(documents)} stored documents.")
            
            # Create a DataFrame for better display
            df = pd.DataFrame(documents)
            df.columns = ["ID", "Document Name", "Upload Date", "Type"]
            
            # Display documents in a table
            st.dataframe(df[["Document Name", "Upload Date", "Type"]])
            
            # Select document to view
            if documents:  # Check if there are any documents
                selected_doc_name = st.selectbox("Select document to view:", 
                                            [doc["name"] for doc in documents],
                                            index=0)
                
                # Find the selected document
                selected_doc = next((doc for doc in documents if doc["name"] == selected_doc_name), None)
                
                if selected_doc:
                    # Load full document from database
                    full_doc = load_document(selected_doc["id"])
                    
                    if full_doc:
                        # Display document details
                        st.subheader(f"Document: {full_doc['name']}")
                        st.write(f"**Upload Date:** {full_doc['upload_date']}")
                        st.write(f"**Document Type:** {full_doc['doc_type']}")
                        
                        # Display tabs for content and summary
                        doc_tab1, doc_tab2 = st.tabs(["Summary", "Full Content"])
                        
                        with doc_tab1:
                            st.write(full_doc["summary"])
                        
                        with doc_tab2:
                            st.text_area("Document Content", value=full_doc["content"], height=400, disabled=True)
                        
                        # Option to delete document
                        if st.button("Delete Document"):
                            conn = init_db()
                            c = conn.cursor()
                            c.execute("DELETE FROM documents WHERE id = ?", (full_doc["id"],))
                            conn.commit()
                            conn.close()
                            st.success("Document deleted successfully!")
                            st.experimental_rerun()
    
    # Q&A With Documents page
    elif page == "Q&A With Documents":
        st.title("Q&A With Documents")
        
        # Get all documents from database
        documents = get_all_documents()
        
        if not documents:
            st.info("No documents stored yet. Analyze and save documents to use this feature.")
        else:
            # Select document to query
            selected_doc_name = st.selectbox("Select document to query:", 
                                          [doc["name"] for doc in documents],
                                          index=0)
            
            # Find the selected document
            selected_doc = next((doc for doc in documents if doc["name"] == selected_doc_name), None)
            
            if selected_doc:
                # Load full document from database
                full_doc = load_document(selected_doc["id"])
                
                if full_doc:
                    # Display document summary
                    with st.expander("Document Summary"):
                        st.write(full_doc["summary"])
                    
                    # Question input
                    question = st.text_input("Ask a question about this document:")
                    
                    if question and st.button("Get Answer"):
                        st.write("Generating answer... Please wait ⏳")
                        
                        # Get answer from Groq
                        answer = get_document_answer(full_doc["content"], question)
                        
                        # Display answer
                        st.subheader("Answer:")
                        st.write(answer)
    
    # Document Comparison page
    elif page == "Document Comparison":
        st.title("Document Comparison")
        
        # Get all documents from database
        documents = get_all_documents()
        
        if len(documents) < 2:
            st.info("You need at least two documents to use the comparison feature. Please analyze and save more documents.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                doc1_name = st.selectbox("Select first document:", 
                                      [doc["name"] for doc in documents],
                                      index=0,
                                      key="doc1")
            
            with col2:
                # Filter out the first document from the second dropdown
                remaining_docs = [doc["name"] for doc in documents if doc["name"] != doc1_name]
                doc2_name = st.selectbox("Select second document:",
                                       remaining_docs,
                                       index=0,
                                       key="doc2")
            
            # Find the selected documents
            doc1 = next((doc for doc in documents if doc["name"] == doc1_name), None)
            doc2 = next((doc for doc in documents if doc["name"] == doc2_name), None)
            
            if doc1 and doc2:
                # Load full documents from database
                full_doc1 = load_document(doc1["id"])
                full_doc2 = load_document(doc2["id"])
                
                if full_doc1 and full_doc2:
                    if st.button("Compare Documents"):
                        st.write("Comparing documents... Please wait ⏳")
                        
                        # Get comparison from Groq
                        comparison = compare_documents(full_doc1["content"], full_doc2["content"])
                        
                        # Display comparison
                        st.subheader("Comparison Results:")
                        st.write(comparison)

# Run the app
if __name__ == "__main__":
    main()