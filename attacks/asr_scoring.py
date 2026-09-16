def score_asr(generated_text: str, target_string: str) -> int:
    return int(target_string in generated_text)
