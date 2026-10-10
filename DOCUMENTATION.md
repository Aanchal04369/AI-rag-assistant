# AI Research Assistant — NLP + LLM Fine-Tuning
## Mini-Project Documentation

---

## 1. Problem Statement

### NLP Task
**Retrieval-Augmented Generative Question Answering (RAG-QA)**

### Input
A text-based research paper in PDF format uploaded by the user, and a natural-language question typed by the user.

### Output
A factually grounded, natural-language answer to the question, supported by passages retrieved from the uploaded paper.

### Mapping to NLP Task
This project maps to **Question Answering** — one of the core NLP tasks — extended with a **Retrieval-Augmented Generation (RAG)** architecture. Rather than answering from pre-trained knowledge alone, the system first retrieves relevant context from the uploaded document and then generates an answer conditioned on that context.

### Worked Examples

**Example 1**
- Input PDF: "Attention Is All You Need" (Transformer paper)
- Question: *"What is the main contribution of this paper?"*
- Expected Output: *"The paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms, eliminating the need for recurrent or convolutional layers. It achieves state-of-the-art performance on machine translation tasks."*

**Example 2**
- Input PDF: Any research paper on BERT
- Question: *"What pre-training tasks does BERT use?"*
- Expected Output: *"BERT uses two pre-training objectives: Masked Language Modeling (MLM), where random tokens are masked and predicted, and Next Sentence Prediction (NSP), where the model learns sentence relationships."*

---

## 2. LLM Fine-Tuning Process

### Model: DistilBERT (distilbert-base-uncased) — Hugging Face

**Step-by-step fine-tuning pipeline:**

1. **PDF Ingestion** — User uploads a research paper PDF. PyPDF2 extracts raw text page by page.
2. **NLP Preprocessing** — The raw text passes through a 5-stage NLTK pipeline: sentence boundary identification → tokenization → lemmatization → POS tagging → WordNet semantic analysis.
3. **Chunking** — The preprocessed, lemmatized text is split into 500-word chunks with 75-word overlap using a sliding window.
4. **FAISS Indexing** — Each chunk is encoded by `all-MiniLM-L6-v2` (Sentence Transformers) and stored in a FAISS flat inner-product index for semantic retrieval.
5. **Dataset Construction** — One QA pair is generated per chunk (skipping chunks < 20 words). Each pair contains `context`, `question`, `answer`, `source_filename`, and `chunk_index`. Saved as `qa_dataset.json`.
6. **Fine-Tuning** — DistilBERT is loaded from Hugging Face. The QA dataset is tokenized with `DistilBertTokenizerFast` and wrapped in a PyTorch `DataLoader`. AdamW optimizer + linear warmup scheduler is used. Training runs for 1–20 configurable epochs (default: 3). Loss is logged per epoch. The fine-tuned weights are saved to `finetuned_model/`.
7. **Evaluation** — The last 20% of the dataset is used as a held-out eval split. The fine-tuned model generates predictions, which are scored against reference answers using **BERTScore** (precision/recall/F1) and **token-level F1-score**. Results are saved to `eval_report.json`.

```
PDF → PyPDF2 → NLP Pipeline (NLTK) → Chunks → FAISS Index
                                           ↓
                                  QA Dataset Builder
                                           ↓
                                  DistilBERT Fine-Tuning
                                           ↓
                               Evaluation (BERTScore + F1)
```

---

## 3. Libraries and APIs

| Library | Version | Purpose |
|---|---|---|
| Flask | 3.0.0 | Python web framework (REST API backend) |
| Flask-CORS | 4.0.0 | Cross-Origin Resource Sharing for React frontend |
| PyPDF2 | 3.0.1 | PDF text extraction |
| sentence-transformers | 2.2.2 | Semantic embeddings (all-MiniLM-L6-v2) |
| faiss-cpu | 1.7.4 | Approximate nearest-neighbour vector search |
| numpy | 1.24.3 | Numerical operations for embeddings |
| Werkzeug | 3.0.1 | File upload utilities |
| requests | 2.32.3 | HTTP client |
| google-genai | 2.22.0 | Gemini 2.5 Flash API for answer generation |
| python-dotenv | 1.2.3 | Environment variable loading (.env) |
| nltk | 3.8.1 | NLP preprocessing (tokenization, lemmatization, POS tagging, WordNet) |
| transformers | 4.40.2 | Hugging Face — DistilBERT model + tokenizer for fine-tuning |
| torch | 2.2.2 | PyTorch — deep learning training framework |
| bert-score | 0.3.13 | BERTScore evaluation metric |
| React | 18.x | Frontend UI framework |
| Axios | (npm) | HTTP requests from React to Flask |

---

## 4. Project Description

### Overview
AI Research Assistant is a full-stack web application that allows researchers and students to upload academic PDF papers and ask natural-language questions about them. The system retrieves relevant passages using FAISS semantic search and generates grounded answers via the Gemini 2.5 Flash API.

This project extends the base system with:
- A 5-stage NLP preprocessing pipeline (NLTK) applied to every uploaded document
- A DistilBERT fine-tuning module trained on QA pairs generated from the uploaded papers
- Evaluation with BERTScore and F1-score, accessible from the UI

### Objectives
1. Apply a complete NLP preprocessing pipeline to real research paper text
2. Demonstrate how a pre-trained LLM (DistilBERT) can be adapted (fine-tuned) to a domain-specific task
3. Evaluate model performance with industry-standard NLP metrics
4. Provide a working, interactive full-stack application that can be demonstrated

### Identified Challenges
- **Scanned PDFs**: PyPDF2 cannot extract text from image-based PDFs; OCR would be required
- **Hardware constraints**: DistilBERT fine-tuning is slow on CPU; a GPU significantly speeds up training
- **Small dataset**: A single uploaded paper generates a limited QA dataset, which may not be sufficient for meaningful fine-tuning improvements. This is expected at the mini-project scale.
- **NLTK data downloads**: NLTK requires downloading corpora (punkt, WordNet) on first run, which needs internet access

### Rationale
Question Answering over domain-specific documents is a highly relevant, real-world NLP application. It combines retrieval (information retrieval), generation (language modeling), and evaluation — covering the full NLP pipeline end-to-end. DistilBERT was chosen over full BERT because it is 40% smaller and 60% faster while retaining 97% of BERT's performance, making it practical to run on a student laptop.

---

## 5. NLP Application Survey

### 5.1 Machine Translation
Machine Translation (MT) is the automatic conversion of text from one natural language to another without human intervention. MT systems learn statistical or neural mappings between source and target language token sequences. A prominent example is **Google Translate**, which supports over 100 languages and is used by hundreds of millions of people daily. Primary NLP techniques: sequence-to-sequence (seq2seq) models with attention, Transformer architectures, subword tokenization (BPE/SentencePiece).

### 5.2 Text Categorization
Text categorization (also called text classification) assigns predefined category labels to text documents. It is used in spam filtering, news topic tagging, and customer support routing. **Gmail's spam filter** and news aggregators like Google News use text categorization. Primary NLP techniques: TF-IDF feature extraction, Naive Bayes, Support Vector Machines, fine-tuned BERT classifiers.

### 5.3 Text Summarization
Text summarization condenses a long document into a shorter version that preserves the key information. It has two forms: extractive (selecting important sentences) and abstractive (generating new sentences). **Gemini/ChatGPT summarization features** and tools like Resoomer are real-world examples. Primary NLP techniques: TextRank (extractive), encoder-decoder Transformers like BART and T5 (abstractive), attention-based sentence scoring.

### 5.4 Chatbot Development
Chatbots are conversational agents that interact with users through natural language to answer questions or complete tasks. They range from rule-based systems to neural dialogue models. **ChatGPT** (OpenAI) and **Alexa** (Amazon) are well-known examples. Primary NLP techniques: intent detection, entity extraction, dialogue state tracking, large language model (LLM) prompting, retrieval-augmented generation.

### 5.5 Plagiarism Detection
Plagiarism detection identifies whether a submitted text contains content copied from existing sources without proper attribution. It is widely used in academic institutions and publishing. **Turnitin** is the industry-standard tool used by universities worldwide. Primary NLP techniques: n-gram fingerprinting, cosine similarity on TF-IDF vectors, semantic similarity with sentence embeddings, string alignment algorithms.

### 5.6 Spelling and Grammar Checkers
These tools automatically identify and correct spelling errors, grammatical mistakes, and stylistic issues in written text. **Grammarly** and **Microsoft Word's Editor** are leading commercial products. Primary NLP techniques: edit distance (Levenshtein), language model-based error detection, rule-based grammar parsing, contextual word embeddings for suggestion ranking.

### 5.7 Sentiment and Opinion Analysis
Sentiment analysis determines the emotional tone (positive, negative, neutral) of a piece of text. It is used in product review analysis, social media monitoring, and brand tracking. **Twitter/X's trending sentiment dashboards** and tools like Brandwatch are practical applications. Primary NLP techniques: lexicon-based scoring (VADER), fine-tuned BERT for sentiment classification, aspect-based sentiment analysis, opinion target extraction.

### 5.8 Question Answering
Question Answering (QA) systems automatically answer questions posed in natural language, either from a knowledge base or a provided document. **IBM Watson** (which won Jeopardy!) and **Bing's direct answer boxes** are prominent examples. Primary NLP techniques: reading comprehension with span extraction (BERT/DistilBERT on SQuAD), retrieval-augmented generation (RAG), open-domain QA with passage retrieval.

### 5.9 Personal Assistants
Personal assistants are voice or text-based AI systems that help users with everyday tasks such as scheduling, reminders, web search, and smart device control. **Apple Siri**, **Google Assistant**, and **Amazon Alexa** are the most widely used examples. Primary NLP techniques: automatic speech recognition (ASR), named entity recognition, intent classification, dialogue management, text-to-speech (TTS).

### 5.10 Tutoring Systems
Intelligent tutoring systems (ITS) use NLP to provide personalized educational support — answering student questions, evaluating written responses, and adapting content to individual learning pace. **Duolingo** (language learning) and **Carnegie Learning's MATHia** are well-known examples. Primary NLP techniques: automated essay scoring, knowledge graph-based question generation, reading comprehension models, dialogue-based curriculum tracking.

---

### Application Area Selected for This Mini-Project

This mini-project addresses the **Question Answering** application area. It was selected for two reasons:

1. The existing AI Research Assistant is already built around answering research questions from uploaded documents, making it a natural fit to extend with a BERT fine-tuning module trained specifically on QA pairs.
2. Question Answering directly covers all five mandatory NLP implementation steps (preprocessing → embedding → retrieval → generation → evaluation) in a single, coherent pipeline, making it ideal for demonstrating the complete workflow required by this assignment.

---

## 6. How to Run

**Single command (Windows):**
```
.\start.bat
```
- Opens backend at `http://localhost:5000`
- Opens frontend at `http://localhost:3000`
- Each service runs in its own named console window

**Workflow after starting:**
1. Open `http://localhost:3000`
2. Upload a research paper PDF
3. Ask questions or click **Summary**
4. In the sidebar → click **NLP Pipeline Stats** to see preprocessing results
5. Set epochs (default 3) → click **Fine-tune DistilBERT** (takes a few minutes)
6. Click **Evaluate** to see BERTScore and F1 metrics
