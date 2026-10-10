# Requirements Document

## Introduction

This feature extends the existing AI Research Assistant (Flask + React, Gemini API, FAISS, Sentence Transformers) with two capabilities:

1. **NLP Preprocessing Pipeline** — a structured text-processing pipeline (sentence boundary detection, tokenization, lemmatization, POS tagging, and WordNet semantic analysis) that integrates into the existing PDF chunking and indexing flow.
2. **LLM Fine-Tuning Module** — the ability to fine-tune a Hugging Face model (BERT or similar) on a domain-specific dataset derived from uploaded research papers, and evaluate the result using BERTScore and F1-score.

Additionally, the feature requires:
- Documentation artifacts (problem statement, fine-tuning process diagram description, libraries list, project description).
- An NLP application survey covering ten application areas.
- A reliable single-command startup that launches both the Flask backend and the React frontend together on Windows.

The feature is **formulation-only at this stage** — requirements describe *what* the system must do, not *how* to implement it. Implementation will follow in a separate phase.

---

## Glossary

- **NLP_Pipeline**: The preprocessing component that transforms raw extracted text through five sequential stages before chunking or indexing.
- **Preprocessor**: The sub-component responsible for a single NLP stage (e.g., tokenizer, lemmatizer).
- **Fine_Tuner**: The component responsible for fine-tuning a pre-trained Hugging Face model on a domain-specific dataset.
- **Evaluator**: The component that computes BERTScore and F1-score for a fine-tuned model.
- **Dataset_Builder**: The component that creates a question-answer training dataset from uploaded research-paper chunks.
- **Startup_Script**: The `start.bat` file (or equivalent) that launches both backend and frontend with a single command.
- **Documentation_Module**: The set of written artifacts describing the project's problem statement, fine-tuning process, libraries, project description, and NLP application survey.
- **System**: The AI Research Assistant application as a whole.
- **Backend**: The Flask Python server running on `localhost:5000`.
- **Frontend**: The React application running on `localhost:3000`.

---

## Requirements

### Requirement 1: NLP Preprocessing Pipeline Integration (REFINED)

**User Story:** As a student, I want the system to apply a structured NLP preprocessing pipeline to every uploaded PDF, so that text entering the FAISS index is linguistically normalized and enriched.

#### Acceptance Criteria

1. WHEN a PDF is uploaded, THE NLP_Pipeline SHALL apply the following five stages to the extracted text in order: sentence boundary identification, tokenization, lemmatization, POS tagging, and WordNet semantic analysis.
2. WHEN sentence boundary identification runs, THE Preprocessor SHALL split the extracted text into individual sentences by treating a period, question mark, or exclamation mark followed by a whitespace character and an uppercase letter as a sentence boundary.
3. WHEN tokenization runs, THE Preprocessor SHALL split each sentence into individual word and punctuation tokens.
4. WHEN lemmatization runs, THE Preprocessor SHALL replace each token with its dictionary base form (lemma).
5. WHEN POS tagging runs, THE Preprocessor SHALL assign a part-of-speech tag to each token.
6. WHEN WordNet semantic analysis runs, THE Preprocessor SHALL attach at least one synset definition from WordNet to each content-bearing token (noun, verb, adjective, adverb) where a synset exists; tokens with no matching synset SHALL be retained unchanged.
7. WHEN the NLP_Pipeline completes, THE System SHALL pass the preprocessed output to the existing chunking function as a single plain-text string of space-separated tokens so that FAISS indexing operates on the pipeline output.
8. IF any single preprocessing stage raises an exception AND at least one prior stage has produced output, THEN THE NLP_Pipeline SHALL log the error with the stage name and continue processing using the output of the last successful stage.
9. IF the first preprocessing stage (sentence boundary identification) raises an exception, THEN THE NLP_Pipeline SHALL log the error and fall back to using the raw extracted text as input to the chunking function.

---

### Requirement 2: Domain-Specific Dataset Construction (REFINED)

**User Story:** As a student, I want the system to build a question-answer dataset from uploaded research papers, so that the fine-tuning process has domain-relevant training data.

#### Acceptance Criteria

1. WHEN a PDF has been processed by the NLP_Pipeline, THE Dataset_Builder SHALL generate exactly one question-answer pair per non-skipped chunk.
2. THE Dataset_Builder SHALL store each generated pair as a JSON object with the fields `context`, `question`, `answer`, `source_filename`, and `chunk_index`.
3. THE Dataset_Builder SHALL write the complete dataset to a designated output file, overwriting any previously generated dataset for the same document.
4. IF a chunk contains fewer than 20 whitespace-delimited words after preprocessing, THEN THE Dataset_Builder SHALL skip that chunk and not generate a pair from it.
5. IF generating a pair for a specific chunk raises an exception, THEN THE Dataset_Builder SHALL log the error with the chunk index and continue processing the remaining chunks.

---

### Requirement 3: LLM Fine-Tuning (REFINED)

**User Story:** As a student, I want to fine-tune a BERT-based model on the domain-specific dataset, so that the model improves its performance on research-paper question answering.

#### Acceptance Criteria

1. WHEN fine-tuning is triggered, THE Fine_Tuner SHALL load a pre-trained BERT-based model from Hugging Face (e.g., `bert-base-uncased` or `distilbert-base-uncased`).
2. WHEN fine-tuning is triggered, THE Fine_Tuner SHALL load the dataset produced by the Dataset_Builder.
3. IF the dataset loaded by the Fine_Tuner contains zero examples, THEN THE Fine_Tuner SHALL abort training and return an error message stating that no training data is available.
4. THE Fine_Tuner SHALL fine-tune the model using the loaded dataset for a configurable number of epochs in the range 1–20 (default: 3).
5. IF the configured epoch count is outside the range 1–20, THEN THE Fine_Tuner SHALL reject the configuration and return an error message identifying the invalid value.
6. WHEN fine-tuning completes, THE Fine_Tuner SHALL save the fine-tuned model weights to a configurable local directory (default name: `finetuned_model`).
7. IF fine-tuning encounters an out-of-memory error, THEN THE Fine_Tuner SHALL reduce the batch size by half and retry once; IF the retry also fails, THEN THE Fine_Tuner SHALL abort and return an error message stating that training failed due to insufficient memory.
8. WHEN each training epoch completes, THE Fine_Tuner SHALL log the epoch number and training loss for that epoch.

---

### Requirement 4: Evaluation with BERTScore and F1-Score (REFINED)

**User Story:** As a student, I want to evaluate the fine-tuned model using BERTScore and F1-score, so that I can measure and report its performance objectively.

#### Acceptance Criteria

1. WHEN evaluation is triggered after fine-tuning, THE Evaluator SHALL compute the BERTScore (precision, recall, F1) for model predictions against reference answers on the last 20% of the dataset (minimum 10 examples) as the held-out evaluation split.
2. WHEN evaluation is triggered after fine-tuning, THE Evaluator SHALL compute the token-level F1-score for model predictions against reference answers, normalizing both prediction and reference to lowercase with punctuation stripped before comparison.
3. WHEN all metrics are computed, THE Evaluator SHALL write the results to a JSON report file with the fields `bertscore_precision`, `bertscore_recall`, `bertscore_f1`, `token_f1`, `model_name`, `example_count`, and `evaluation_timestamp` (ISO 8601 UTC).
4. IF the evaluation dataset contains fewer than 10 examples, THEN THE Evaluator SHALL log a warning stating that the sample size is too small for reliable metrics AND SHALL proceed with evaluation rather than aborting.
5. WHEN evaluation completes, THE Evaluator SHALL include the computed metrics in the Backend API response so that the Frontend can render them to the user.
6. IF the evaluation process raises an exception, THEN THE Evaluator SHALL return an error message identifying the failure reason and SHALL NOT write a partial report file.

---

### Requirement 5: Documentation Artifacts (REFINED)

**User Story:** As a student, I want the project to include required documentation artifacts, so that the mini-project submission is complete.

#### Acceptance Criteria

1. THE Documentation_Module SHALL include a Problem Statement document that defines the input (research-paper PDFs) and the output (answers to user questions), and maps the task to the NLP task category of Retrieval-Augmented Generative Question Answering.
2. THE Documentation_Module SHALL include at least two worked examples in the Problem Statement, each showing an input question and an output answer grounded in an actual uploaded document's content.
3. THE Documentation_Module SHALL include a pipeline description structured as a numbered sequence of at least five steps covering the full flow from raw PDF ingestion to LLM-generated answer (extraction → chunking → embedding → FAISS indexing → retrieval → generation).
4. THE Documentation_Module SHALL include a Libraries and APIs list that names every library present in the project's `requirements.txt` file at its exact pinned version.
5. THE Documentation_Module SHALL include a Project Description covering: overview, objectives, at least one identified challenge, and rationale for the chosen NLP task and model.

---

### Requirement 6: NLP Application Survey (REFINED)

**User Story:** As a student, I want a documented survey of NLP applications, so that I can demonstrate awareness of the broader NLP landscape and justify my project's application area.

#### Acceptance Criteria

1. THE Documentation_Module SHALL contain a survey section that covers all ten of the following NLP application areas: Machine Translation, Text Categorization, Text Summarization, Chatbot Development, Plagiarism Detection, Spelling and Grammar Checking, Sentiment and Opinion Analysis, Question Answering, Personal Assistants, and Tutoring Systems.
2. WHILE the survey section is being reviewed, FOR EACH of the ten application areas THE Documentation_Module SHALL include: a description of at least three sentences, one real-world example product or system, and at least two primary NLP techniques involved.
3. THE Documentation_Module SHALL include a concluding paragraph that explicitly names the application area this mini-project addresses and provides at least two distinct reasons tied to the project's scope or features explaining why it was selected.

---

### Requirement 7: Single-Command Startup (REFINED)

**User Story:** As a student, I want to start both the backend and frontend with a single command on Windows, so that I do not need to manage two terminal windows manually during development.

#### Acceptance Criteria

1. THE Startup_Script SHALL start the Flask Backend on `localhost:5000` and the React Frontend on `localhost:3000` from a single invocation.
2. WHEN the Startup_Script is run, THE Startup_Script SHALL verify that Python 3.10 or later and Node.js 18 or later are present on PATH before proceeding; version detection SHALL check the actual version number, not merely binary presence.
3. WHEN the Startup_Script starts the Backend, THE Startup_Script SHALL open the Backend in a separate named console window using `cmd /k` so that its log output remains visible even if the process exits.
4. WHEN the Startup_Script starts the Frontend, THE Startup_Script SHALL open the Frontend in a separate named console window using `cmd /k` so that its log output remains visible even if the process exits.
5. THE Startup_Script SHALL NOT check for or require Ollama to be installed; any existing Ollama check SHALL be removed because the Backend uses the Gemini API.
6. IF Python is not found on PATH, THEN THE Startup_Script SHALL print a message stating "Python 3.10 or later is required but was not found on PATH" and exit without starting either service.
7. IF Node.js is not found on PATH, THEN THE Startup_Script SHALL print a message stating "Node.js 18 or later is required but was not found on PATH" and exit without starting either service.
8. IF `pip install -r requirements.txt` or `npm install` exits with a non-zero code, THEN THE Startup_Script SHALL print a message identifying which dependency installation failed and exit without starting either service.

---

### Requirement 8: Round-Trip Text Fidelity (REFINED)

**User Story:** As a student, I want the NLP pipeline output to faithfully reconstruct the original token sequence, so that no information is silently dropped during preprocessing.

#### Acceptance Criteria

1. WHEN the tokenization stage completes, THE Preprocessor SHALL produce a token list such that the multiset of characters in the joined token string equals the multiset of non-whitespace characters in the original input (every character and its count must be accounted for).
2. WHEN any preprocessing stage produces output, THE Preprocessor SHALL include both the original surface-form token text and the normalized lemma in each output token record.
3. IF the tokenization stage produces zero tokens from a non-empty input, THEN THE NLP_Pipeline SHALL log an error and fall back to treating the entire input string as a single token to prevent data loss.
