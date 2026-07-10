"""
Schema normalization across the three corpora.

nq_open, hotpot_qa, and ms_marco each expose different field names for the
question, the gold answer(s), and the supporting passages. The mappings
below are best-effort based on each corpus's documented schema. The FIRST
time you load each corpus, run:

    from data.loader import load_corpus
    recs = load_corpus("nq_open", split="dev")
    print(recs[0].keys())

...and confirm the field names below actually match what comes back. HF
dataset schemas do shift between versions, and it costs five minutes to
check versus a baseline run silently reading the wrong field for weeks.
"""


def extract_passage_text(corpus_name: str, record: dict) -> str:
    """Return the text to embed for a single record (used when building the FAISS index)."""
    if corpus_name == "nq_open":
        return record.get("question", "") + " " + " ".join(record.get("answer", []))
    if corpus_name == "hotpot_qa":
        # hotpot_qa's "context" is a list of [title, sentences] pairs in the distractor config
        context = record.get("context", {})
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        flat = []
        for t, s in zip(titles, sentences):
            flat.append(t + ": " + " ".join(s))
        return " ".join(flat) if flat else record.get("question", "")
    if corpus_name == "ms_marco":
        passages = record.get("passages", {})
        texts = passages.get("passage_text", [])
        return " ".join(texts) if texts else record.get("query", "")
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")


def extract_question(corpus_name: str, record: dict) -> str:
    if corpus_name == "nq_open":
        return record.get("question", "")
    if corpus_name == "hotpot_qa":
        return record.get("question", "")
    if corpus_name == "ms_marco":
        return record.get("query", "")
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")


def extract_gold_answers(corpus_name: str, record: dict) -> list:
    if corpus_name == "nq_open":
        return record.get("answer", [])
    if corpus_name == "hotpot_qa":
        ans = record.get("answer")
        return [ans] if ans else []
    if corpus_name == "ms_marco":
        answers = record.get("answers", [])
        return answers if isinstance(answers, list) else [answers]
    raise ValueError(f"No normalization rule for corpus '{corpus_name}'")
