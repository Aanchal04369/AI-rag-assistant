# AI Research Assistant — Local RAG + LLM

A research-paper assistant that uploads a text-based PDF, creates semantic embeddings with Sentence Transformers + FAISS, retrieves relevant passages, and generates grounded answers and summaries with a local Ollama LLM.

## Architecture

PDF → PyPDF2 → chunks → Sentence Transformers → FAISS → relevant passages → Ollama/Llama 3.2 → answer/summary

## One-time setup on Windows

1. Install Python 3.10+ and Node.js 18+.
2. Install Ollama for Windows: https://ollama.com/download/windows
3. Open a new PowerShell/Command Prompt and run:

```text
ollama pull llama3.2
```

This downloads the local Llama 3.2 model. You only need to do it once.

## Run

Open the project folder in VS Code and run:

```powershell
.\start.bat
```

Then open `http://localhost:3000`.

## How to use

1. Click **Create New Chat**.
2. Upload a text-based PDF (maximum 16 MB).
3. Wait for processing.
4. Click **Summary** for an AI-generated research-paper summary.
5. Or ask questions such as:
   - What is the main contribution of this paper?
   - What methodology was used?
   - What are the key findings?
   - What are the limitations?

## Privacy

The PDF, embeddings, retrieval, and LLM generation run locally. The application does not send the paper to Gemini, OpenAI, Groq, or another cloud LLM.

## Important limitation

The current PDF extractor is text-based. Scanned/image-only PDFs need OCR before they can be searched.
