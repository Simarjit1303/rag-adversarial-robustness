"""
Attack 1 of 3 (Phase 2): indirect prompt injection.

Differs from PoisonedRAG (a wrong FACT planted in the corpus, forcibly
retrieved into top-k) by testing a genuinely different attack surface: a
wrong INSTRUCTION planted in a document that is retrieved ORGANICALLY --
this attack never touches which documents get retrieved, only what one
already-retrieved document's text says. Naming and structure throughout
this module are kept distinct from any PoisonedRAG-style code for exactly
this reason -- these are testing different attack surfaces, not two
flavors of "put bad text in the corpus."

Injection point: the rank-1 (highest-scoring) organically retrieved
document. Chosen over a random rank among top-k because it is the most
defensible "organic" placement -- an attacker who wants their tampered
content to matter needs it to be a document that would be retrieved
regardless of the attack, and rank-1 is the strongest such candidate
without any change to retrieval itself. Deterministic given
(question, corpus, template): no randomness, so results are exactly
reproducible.

nq_open is excluded here (unfixable gold-answer leakage -- see
docs/nq_open_leakage_finding.md and config.py's CORPORA comment). This attack
runs against hotpot_qa and ms_marco only; see
evaluation/result_paths.py's ATTACK_ELIGIBLE_CORPORA.
"""

import torch

from attacks.injection_templates import TEMPLATES
from data.build_index import build_index, retrieve
from data.normalize import extract_passage_text
from harness.model_loader import build_chat_prompt
from harness.pipeline import SYSTEM_PROMPT, clean_generation


def build_attack_user_prompt(index, records, question: str, corpus_name: str,
                              injection_template: str, top_k: int = 5):
    """
    Mirrors harness.pipeline.build_rag_user_prompt's context rendering, but
    replaces the rank-1 retrieved document's rendered text with its
    injected version. Every other retrieved document renders exactly as
    the clean baseline does (extract_passage_text, same as
    harness/pipeline.py) -- only the one slot differs, so a delta against
    the Phase 1 baseline isolates the injection's effect, not a change in
    how untouched documents are shown.

    Returns (user_prompt, retrieved, target_string, hijack_type) --
    target_string/hijack_type come from the template, so callers can score
    ASR without re-looking up the template.
    """
    if injection_template not in TEMPLATES:
        raise ValueError(
            f"Unknown injection template '{injection_template}'. "
            f"Options: {list(TEMPLATES)}"
        )
    template = TEMPLATES[injection_template]

    retrieved = retrieve(index, records, question, k=top_k)
    if not retrieved:
        raise ValueError("No documents retrieved -- cannot inject into an empty top-k.")

    lines = []
    for i, (doc, score) in enumerate(retrieved):
        text = extract_passage_text(corpus_name, doc)
        if i == 0:
            text = template.render(text)
        lines.append(f"[{i + 1}] {text}")
    context = "\n\n".join(lines)

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return user_prompt, retrieved, template.target_string, template.hijack_type


def run_attack_query(model, tokenizer, model_key: str, corpus_name: str, question: str,
                      injection_template: str, index=None, records=None,
                      top_k: int = 5, max_new_tokens: int = 256):
    """
    Runs one question through retrieval + injection + generation. Mirrors
    harness.pipeline.run_query's HF-path shape exactly (same fields on the
    returned dict, plus injection-specific ones) so evaluation code can
    reuse the same scoring calls (evaluation.metrics) for utility, with
    attacks.asr_scoring.score_asr layered on top for attack success.
    """
    if index is None or records is None:
        index, records = build_index(corpus_name, split="dev")

    user_prompt, retrieved, target_string, hijack_type = build_attack_user_prompt(
        index, records, question, corpus_name, injection_template, top_k=top_k
    )
    prompt = build_chat_prompt(model_key, tokenizer, SYSTEM_PROMPT, user_prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for reproducibility, same as Phase 1
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    generated = generated.strip()

    return {
        "question": question,
        "retrieved_doc_ids": [idx for idx, _ in enumerate(retrieved)],
        "retrieval_scores": [score for _, score in retrieved],
        "generated_answer": generated,
        "generated_answer_clean": clean_generation(generated),
        "injection_template": injection_template,
        "hijack_type": hijack_type,
        "target_string": target_string,
    }
