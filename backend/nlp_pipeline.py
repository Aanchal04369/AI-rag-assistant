"""
NLP Preprocessing Pipeline
Stages: Sentence Boundary → Tokenization → Lemmatization → POS Tagging → WordNet
"""

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy NLTK bootstrap — download only what is needed
# ---------------------------------------------------------------------------

def _ensure_nltk():
    import nltk
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("taggers/averaged_perceptron_tagger",    "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng","averaged_perceptron_tagger_eng"),
        ("corpora/wordnet",           "wordnet"),
        ("corpora/omw-1.4",           "omw-1.4"),
        ("corpora/stopwords",         "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Stage 1 – Sentence Boundary Identification
# ---------------------------------------------------------------------------

def sentence_boundary_identification(text: str) -> list[str]:
    """Split text into sentences using NLTK sent_tokenize."""
    try:
        _ensure_nltk()
        import nltk
        sentences = nltk.sent_tokenize(text)
        logger.info(f"[NLP] Stage 1 – {len(sentences)} sentences identified.")
        return sentences
    except Exception as exc:
        logger.error(f"[NLP] Stage 1 (sentence boundary) failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Stage 2 – Tokenization
# ---------------------------------------------------------------------------

def tokenize(sentences: list[str]) -> list[list[str]]:
    """Tokenize each sentence into word/punctuation tokens."""
    try:
        _ensure_nltk()
        import nltk
        tokenized = [nltk.word_tokenize(s) for s in sentences]
        total = sum(len(t) for t in tokenized)
        logger.info(f"[NLP] Stage 2 – {total} tokens produced.")
        return tokenized
    except Exception as exc:
        logger.error(f"[NLP] Stage 2 (tokenization) failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Stage 3 – Lemmatization
# ---------------------------------------------------------------------------

def lemmatize(tokenized_sentences: list[list[str]]) -> list[list[dict]]:
    """
    Replace each token with its lemma.
    Returns list of sentences; each sentence is a list of
    {"surface": original_token, "lemma": lemma_form}.
    """
    try:
        _ensure_nltk()
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()

        result = []
        for sent in tokenized_sentences:
            result.append([
                {"surface": tok, "lemma": lem.lemmatize(tok.lower())}
                for tok in sent
            ])
        logger.info("[NLP] Stage 3 – Lemmatization complete.")
        return result
    except Exception as exc:
        logger.error(f"[NLP] Stage 3 (lemmatization) failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Stage 4 – POS Tagging
# ---------------------------------------------------------------------------

def pos_tag(lemmatized_sentences: list[list[dict]]) -> list[list[dict]]:
    """
    Add POS tag to each token record.
    Each dict gets a new key "pos".
    """
    try:
        _ensure_nltk()
        import nltk

        result = []
        for sent in lemmatized_sentences:
            surfaces = [t["surface"] for t in sent]
            tagged   = nltk.pos_tag(surfaces)
            enriched = []
            for token_dict, (_, tag) in zip(sent, tagged):
                enriched.append({**token_dict, "pos": tag})
            result.append(enriched)

        logger.info("[NLP] Stage 4 – POS tagging complete.")
        return result
    except Exception as exc:
        logger.error(f"[NLP] Stage 4 (POS tagging) failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Stage 5 – Semantic Analysis (WordNet)
# ---------------------------------------------------------------------------

_CONTENT_POS = {"NN", "NNS", "NNP", "NNPS",   # Nouns
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs
                "JJ", "JJR", "JJS",             # Adjectives
                "RB", "RBR", "RBS"}             # Adverbs

def _penn_to_wn(tag: str) -> str | None:
    """Convert Penn POS tag to WordNet POS constant."""
    from nltk.corpus import wordnet as wn
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("R"): return wn.ADV
    return None


def wordnet_analysis(pos_tagged_sentences: list[list[dict]]) -> list[list[dict]]:
    """
    Attach the first WordNet synset definition to content-bearing tokens.
    Non-content tokens or tokens with no synset are retained unchanged.
    """
    try:
        _ensure_nltk()
        from nltk.corpus import wordnet as wn

        result = []
        for sent in pos_tagged_sentences:
            enriched = []
            for token_dict in sent:
                wn_pos = _penn_to_wn(token_dict.get("pos", ""))
                synset_def = None
                if wn_pos:
                    synsets = wn.synsets(token_dict["lemma"], pos=wn_pos)
                    if synsets:
                        synset_def = synsets[0].definition()
                enriched.append({**token_dict, "synset": synset_def})
            result.append(enriched)

        logger.info("[NLP] Stage 5 – WordNet semantic analysis complete.")
        return result
    except Exception as exc:
        logger.error(f"[NLP] Stage 5 (WordNet) failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_text: str) -> dict:
    """
    Run all 5 NLP stages on raw_text.

    Returns:
        {
          "plain_text":   str,          # space-joined lemmas → fed to chunk_text()
          "token_records": list[dict],  # flat list of all enriched token dicts
          "sentence_count": int,
          "token_count": int,
          "stages_completed": list[str]
        }
    """
    stages_completed = []
    last_good = None   # track fallback state

    # ---- Stage 1 ----
    try:
        sentences = sentence_boundary_identification(raw_text)
        stages_completed.append("sentence_boundary")
        last_good = "stage1"
    except Exception:
        logger.warning("[NLP] Falling back to raw text (stage 1 failed).")
        # Split on newlines as minimal fallback
        sentences = [s for s in raw_text.split("\n") if s.strip()] or [raw_text]

    # ---- Stage 2 ----
    try:
        tokenized = tokenize(sentences)
        stages_completed.append("tokenization")
        last_good = "stage2"
    except Exception:
        logger.warning("[NLP] Falling back to whitespace tokenization.")
        tokenized = [s.split() for s in sentences]

    # ---- Stage 3 ----
    try:
        lemmatized = lemmatize(tokenized)
        stages_completed.append("lemmatization")
        last_good = "stage3"
    except Exception:
        logger.warning("[NLP] Skipping lemmatization.")
        lemmatized = [[{"surface": t, "lemma": t} for t in sent] for sent in tokenized]

    # ---- Stage 4 ----
    try:
        pos_tagged = pos_tag(lemmatized)
        stages_completed.append("pos_tagging")
        last_good = "stage4"
    except Exception:
        logger.warning("[NLP] Skipping POS tagging.")
        pos_tagged = [[{**t, "pos": "NN"} for t in sent] for sent in lemmatized]

    # ---- Stage 5 ----
    try:
        enriched = wordnet_analysis(pos_tagged)
        stages_completed.append("wordnet")
    except Exception:
        logger.warning("[NLP] Skipping WordNet analysis.")
        enriched = [[{**t, "synset": None} for t in sent] for sent in pos_tagged]

    # ---- Flatten to plain text ----
    flat_tokens = [tok for sent in enriched for tok in sent]

    # Zero-token safety guard
    if not flat_tokens and raw_text.strip():
        logger.error("[NLP] Zero tokens produced — using raw text as single token.")
        flat_tokens = [{"surface": raw_text, "lemma": raw_text, "pos": "NN", "synset": None}]

    plain_text = " ".join(t["lemma"] for t in flat_tokens)

    return {
        "plain_text":      plain_text,
        "token_records":   flat_tokens,
        "sentence_count":  len(sentences),
        "token_count":     len(flat_tokens),
        "stages_completed": stages_completed,
    }
