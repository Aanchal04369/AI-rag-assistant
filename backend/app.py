from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from werkzeug.utils import secure_filename

from nlp_pipeline import run_pipeline
from fine_tuner import build_dataset, fine_tune, evaluate

# ── Pre-download NLTK data at startup ─────────────────────────────────
import nltk
import logging
_nltk_logger = logging.getLogger("nltk_setup")
for _pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception as _e:
        print(f"[NLTK] Could not download {_pkg}: {_e}")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- GEMINI SETUP ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env")
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini API configured.")

# ---------------- EMBEDDING MODEL ----------------

print("Loading local embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Stores the currently uploaded document
document_store = {
    "chunks": [],
    "index": None,
    "metadata": [],
    "filename": None
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_pdf(path):
    text = []

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page_no, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    text.append(
                        f"[Page {page_no}]\n{page_text}"
                    )

    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {exc}")

    return "\n\n".join(text)


def chunk_text(text, chunk_size=500, overlap=75):
    words = text.split()

    if not words:
        return []

    step = max(1, chunk_size - overlap)

    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(
            words[start:start + chunk_size]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def build_index(chunks):
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


def retrieve(query, k=6):
    q = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q)

    k = min(k, len(document_store["chunks"]))

    scores, indices = document_store["index"].search(
        q,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": document_store["chunks"][int(idx)],
                "relevance_score": float(score),
                "metadata": document_store["metadata"][int(idx)]
            })

    return results


def context_from_results(results):
    blocks = []

    for i, item in enumerate(results, 1):
        page = item["metadata"].get(
            "page",
            "unknown"
        )

        blocks.append(
            f"[Source {i} | {page}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(blocks)


# ---------------- GEMINI FUNCTION ----------------

def call_gemini(prompt, temperature=0.2):
    if client is None:
        raise RuntimeError(
            "Gemini API key is missing. "
            "Add GEMINI_API_KEY to backend/.env"
        )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature
            )
        )

    except Exception as exc:
        raise RuntimeError(
            f"Gemini API error: {exc}"
        )

    answer = (response.text or "").strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer


# ---------------- HEALTH ----------------

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gemini": client is not None,
        "model": GEMINI_MODEL
    })


# ---------------- UPLOAD PDF ----------------

@app.post("/api/upload")
def upload():
    global document_store

    if "file" not in request.files:
        return jsonify({
            "error": "No PDF was provided."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "error": "No file was selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only PDF files are supported."
        }), 400

    filename = secure_filename(file.filename)

    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:
        file.save(path)

        # Extract text
        text = extract_text_from_pdf(path)

        if not text.strip():
            return jsonify({
                "error": (
                    "No readable text was found in this PDF. "
                    "Scanned PDFs need OCR."
                )
            }), 400

        # ── NLP Preprocessing Pipeline ──────────────────────────────
        try:
            pipeline_result = run_pipeline(text)
            processed_text  = pipeline_result["plain_text"]
            pipeline_stats  = {
                "sentence_count":   pipeline_result["sentence_count"],
                "token_count":      pipeline_result["token_count"],
                "stages_completed": pipeline_result["stages_completed"],
            }
        except Exception as nlp_exc:
            print(f"[NLP] Pipeline failed, using raw text: {nlp_exc}")
            processed_text = text
            pipeline_stats = {"sentence_count": 0, "token_count": 0, "stages_completed": [], "warning": str(nlp_exc)}

        # Split processed text into chunks
        chunks = chunk_text(processed_text)

        # Create FAISS vector index
        index = build_index(chunks)

        # Build QA dataset for fine-tuning
        build_dataset(chunks, filename)

        document_store = {
            "chunks": chunks,
            "index": index,
            "metadata": [
                {
                    "filename": filename,
                    "chunk_id": i,
                    "page": f"chunk_{i}"
                }
                for i in range(len(chunks))
            ],
            "filename": filename,
            "pipeline_stats": pipeline_stats,
        }

        return jsonify({
            "message":       "PDF processed successfully.",
            "filename":      filename,
            "chunks_count":  len(chunks),
            "nlp_pipeline":  pipeline_stats,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500

    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------- ASK QUESTION ----------------

@app.post("/api/query")
def query():
    if document_store["index"] is None:
        return jsonify({
            "error": "Upload a PDF first."
        }), 400

    data = request.get_json(
        silent=True
    ) or {}

    question = (
        data.get("question") or ""
    ).strip()

    history = data.get("history") or []

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    try:
        # Retrieve relevant chunks using FAISS
        results = retrieve(
            question,
            6
        )

        context = context_from_results(
            results
        )

        # Previous conversation
        history_text = ""

        for item in history[-6:]:
            role = item.get("role")
            text = item.get("text")

            if role in {"user", "assistant"} and text:
                history_text += (
                    f"{role.upper()}: "
                    f"{text[:4000]}\n"
                )

        prompt = f"""
You are an AI Research Assistant.

You are answering questions about an uploaded research paper.

IMPORTANT RULES:
1. Use ONLY the supplied research-paper context.
2. Do not invent facts.
3. If the answer is not supported by the context, say:
   "The information is not available in the retrieved passages."
4. Give a clear and useful answer.
5. When possible, mention the relevant page/source.
6. Use the conversation history only to understand the user's question.

RESEARCH PAPER CONTEXT:

{context}

CONVERSATION HISTORY:

{history_text}

USER QUESTION:

{question}

Answer the question based strictly on the research-paper context.
"""

        answer = call_gemini(
            prompt,
            temperature=0.2
        )

        return jsonify({
            "answer": answer,
            "relevant_chunks": results[:4],
            "filename": document_store["filename"],
            "model": GEMINI_MODEL
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 503


# ---------------- SUMMARY ----------------

@app.post("/api/summary")
def summary():
    if document_store["index"] is None:
        return jsonify({
            "error": "Upload a PDF first."
        }), 400

    try:
        chunks = document_store["chunks"]

        # Select chunks across the whole document
        if len(chunks) <= 14:
            selected = list(
                enumerate(chunks)
            )
        else:
            positions = np.linspace(
                0,
                len(chunks) - 1,
                14,
                dtype=int
            )

            selected = [
                (
                    int(i),
                    chunks[int(i)]
                )
                for i in positions
            ]

        context = "\n\n".join(
            f"[Section {i + 1}]\n{text}"
            for i, text in selected
        )

        prompt = f"""
You are an AI research-paper summarization assistant.

Summarize the supplied research paper text.

IMPORTANT RULES:
1. Use ONLY the supplied paper text.
2. Do not invent information.
3. Keep the summary academically accurate.
4. Clearly explain the paper in simple but professional language.

Use these headings:

## Overview
Explain what the paper is about and its main contribution.

## Methodology
Explain the main approach, architecture, techniques, or methods used.

## Key Findings
Explain the important results, observations, or advantages.

## Conclusion
Explain the main conclusion and significance of the work.

RESEARCH PAPER TEXT:

{context}
"""

        answer = call_gemini(
            prompt,
            temperature=0.15
        )

        return jsonify({
            "answer": answer,
            "filename": document_store["filename"],
            "model": GEMINI_MODEL
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 503


# ---------------- RESET ----------------

@app.post("/api/reset")
def reset():
    global document_store

    document_store = {
        "chunks": [],
        "index": None,
        "metadata": [],
        "filename": None
    }

    return jsonify({
        "message": "Reset complete."
    })


# ---------------- NLP PIPELINE PREVIEW ----------------

@app.post("/api/nlp-preview")
def nlp_preview():
    """Run NLP pipeline on a short text snippet and return enriched tokens."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Provide a 'text' field."}), 400

    try:
        result = run_pipeline(text[:2000])   # cap preview to 2000 chars
        # Return first 30 tokens to keep the response small
        preview_tokens = result["token_records"][:30]
        return jsonify({
            "sentence_count":   result["sentence_count"],
            "token_count":      result["token_count"],
            "stages_completed": result["stages_completed"],
            "sample_tokens":    preview_tokens,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------- FINE-TUNE ----------------

@app.post("/api/finetune")
def finetune():
    """Fine-tune DistilBERT on the QA dataset built from the uploaded PDF."""
    data   = request.get_json(silent=True) or {}
    epochs = data.get("epochs", 3)

    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        return jsonify({"error": "epochs must be an integer."}), 400

    result = fine_tune(epochs=epochs)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


# ---------------- EVALUATE ----------------

@app.post("/api/evaluate")
def evaluate_model():
    """Evaluate the fine-tuned model with BERTScore + F1."""
    result = evaluate()

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


# ---------------- PIPELINE STATUS ----------------

@app.get("/api/pipeline-status")
def pipeline_status():
    """Return NLP pipeline stats for the currently loaded document."""
    if document_store["index"] is None:
        return jsonify({"error": "No document loaded."}), 400

    return jsonify({
        "filename":      document_store["filename"],
        "chunks_count":  len(document_store["chunks"]),
        "pipeline_stats": document_store.get("pipeline_stats", {}),
    })


# ---------------- START SERVER ----------------

if __name__ == "__main__":
    print(
        f"Starting AI Research Assistant on "
        f"http://localhost:5000 | "
        f"LLM: Gemini ({GEMINI_MODEL})"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )