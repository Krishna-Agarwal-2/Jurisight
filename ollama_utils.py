import requests
import time
import streamlit as st
from typing import Dict, Any

OLLAMA_API_URL = "http://localhost:11434/api/generate"
#For Docker, use "http://host.docker.internal:11434/api/generate"
MAX_RETRIES = 5
RETRY_DELAY = 2
MODEL_NAME = "llama3.2:latest"  # Updated to exact model name

def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def ask_ollama_system_prompt(document_text: str, task_prompt: str) -> str:
    """Send document with system prompt to local Ollama instance."""
    if not check_ollama_connection():
        return """Error: Cannot connect to Ollama. Please ensure:
        1. Ollama is running in PowerShell
        2. PowerShell command 'ollama run llama3.2:latest' is active
        3. Port 11434 is not blocked"""

    prompt = f"""<s>[INST] You are a legal document analyzer. 
    Based on the task and document below, provide a detailed analysis.
    
    Task: {task_prompt}
    
    Document:
    {document_text}
    [/INST]</s>"""

    payload = {
        "model": MODEL_NAME,  # Using exact model name from ollama list
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "num_ctx": 4096  # Added context window size
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            with st.spinner(f"💭 Analyzing with {MODEL_NAME}... (Attempt {attempt + 1}/{MAX_RETRIES})"):
                response = requests.post(OLLAMA_API_URL, json=payload)
                
                if response.status_code == 404:
                    error_msg = response.json().get('error', '').lower()
                    if 'model not found' in error_msg:
                        return f"Error: Model '{MODEL_NAME}' not found. Available models: {error_msg}"
                    return "Error: Cannot connect to Ollama API endpoint"
                    
                response.raise_for_status()
                result = response.json()
                if 'response' in result:
                    return result['response']
                return "Error: No response generated"
                
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                return "Error: Cannot connect to Ollama. Is it running in PowerShell?"
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"Error: Failed after {MAX_RETRIES} attempts - {str(e)}"
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return "Error: Failed to get response after multiple attempts"