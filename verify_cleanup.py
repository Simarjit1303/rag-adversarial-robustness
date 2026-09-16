import ast
import re
import string
from collections import Counter
qwen_samples = [('Mercedes-Benz Stadium', 'The Chick-fil-A Kickoff Game is being played at Mercedes-Benz Stadium.'), ('0.71% of the population', 'The incarceration rate in the United States is 0.71% of the population.'), ('Six Flags Magic Mountain', 'Six Flags Magic Mountain holds the record for the most rides.'), ('Nanak', 'The founder and the first of the ten gurus of Sikhism is Nanak.'), ('Ottaviano Petrucci', 'The first person to master the art of printing polyphonic music was Ottaviano Petrucci.'), ('acetabulum', 'The ilium, ischium, and pubis meet at the acetabulum.'), ('1982', 'The context provided states that "Party Like It 1999" came out in 1982.'), ('Selena Gomez', 'The most followed person on Inst is Selena Gomez.'), ('Katniss Everdeen', 'Jennifer Lawrence plays Katniss Everdeen in *The Hunger Games*.'), ('Washington Redskins', 'In 2017, Kirk Cousins played for the Washington Redskins.'), ('drowned', 'Beverly died by drowning in August Osage County.'), ('Channel 150', 'The answer is Channel 150.'), ('carnivore', 'A T. rex is a carnivore.'), ('The asteroid belt', 'The asteroid belt.'), ('President Abraham Lincoln', 'The answer is President Abraham Lincoln.'), ('Alan Shearer', 'Alan Shearer scored the most goals in the Premier League.'), ('Geoff Stults', 'Geoff Stults'), ('The Emancipation of Mimi', 'The album "We Belong Together" by Mariah Carey is on *The Emancipation of Mimi*.'), ('1519', 'The Spanish conquest of the Aztecs began in 1519.'), ('1999', 'Angel left *Buffy the Vampire Slayer* in 1999.')]
ministral_samples = [('Mercedes-Benz Stadium', 'Mercedes-Benz Stadium'), ('0.71% of the population', 'The incarceration rate in the United States is **0.71% of the population**.'), ('Six Flags Magic Mountain', 'Six Flags Magic Mountain holds the record for most rides.'), ('Nanak', 'Nanak'), ('Ottaviano Petrucci', 'Ottaviano Petrucci'), ('acetabulum', 'The ilium, ischium, and pubis meet at the **acetabulum**.'), ('1982', 'The context states that **"Party Like It\'s 1999"** came out in **1982**.'), ('Selena Gomez', 'Selena Gomez'), ('Katniss Everdeen', 'Jennifer Lawrence plays **Katniss Everdeen** in *The Hunger Games*.'), ('Washington Redskins', 'Kirk Cousins played for the **Washington Redskins** in 2017.'), ('drowned', 'Beverly died by **drowning**.'), ('Channel 150', 'Channel 150'), ('carnivore', 'A T. rex is a **carnivore**.'), ('The asteroid belt', 'The asteroid belt.'), ('President Abraham Lincoln', 'President Abraham Lincoln.'), ('Alan Shearer', 'Alan Shearer scored the most goals in the Premier League.'), ('Geoff Stults', 'Geoff Stults plays Leif on *Man with a Plan*.'), ('The Emancipation of Mimi', 'The song *"We Belong Together"* by Mariah Carey is on the album **The Emancipation of Mimi**.'), ('1519', 'The Spanish conquest of the Aztecs began in **1519**.'), ('1999', 'Angel left *Buffy the Vampire Slayer* in **1999**.')]

def squad_normalize(s):
    s = s.lower()
    s = re.sub('\\b(a|an|the)\\b', ' ', s)
    s = ''.join((ch for ch in s if ch not in set(string.punctuation)))
    return ' '.join(s.split())

def exact_match(pred, gold):
    return int(squad_normalize(pred) == squad_normalize(gold))

def f1(pred, gold):
    pred_toks, gold_toks = (squad_normalize(pred).split(), squad_normalize(gold).split())
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    p, r = (num_same / len(pred_toks), num_same / len(gold_toks))
    return 2 * p * r / (p + r)

def contains_answer(pred, gold):
    return int(squad_normalize(gold) in squad_normalize(pred))
PREAMBLE_PATTERNS = ['^the answer is[:\\s]*', '^the correct answer is[:\\s]*', '^according to (the )?(context|passage|document)[,:]?\\s*', '^the context (provided |states|indicates|says)[^,\\.]*(states|indicates|says|that)[,:]?\\s*']

def _try_extract_leaked_answer(text):
    dict_match = re.search("\\{.*'answer':\\s*\\[.*?\\].*\\}", text)
    if dict_match:
        try:
            parsed = ast.literal_eval(dict_match.group(0))
            answer = parsed.get('answer')
            if isinstance(answer, list) and answer:
                return str(answer[0])
        except (ValueError, SyntaxError):
            pass
    stripped = text.strip()
    if re.match('^\\[\\s*[\'\\"]', stripped):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return None

def clean_generation(text):
    extracted = _try_extract_leaked_answer(text)
    if extracted is not None:
        text = extracted
    text = re.sub('\\*\\*(.*?)\\*\\*', '\\1', text)
    text = re.sub('\\*(.*?)\\*', '\\1', text)
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub('</?think>', '', text)
    t = text.strip()
    for pat in PREAMBLE_PATTERNS:
        new_t = re.sub(pat, '', t, flags=re.IGNORECASE)
        if new_t != t:
            t = new_t.strip()
            break
    return t.rstrip('.').strip()

def check(samples, label, expected_em_raw, expected_em_clean, expected_contains):
    em_raw = sum((exact_match(raw, gold) for gold, raw in samples))
    em_clean = sum((exact_match(clean_generation(raw), gold) for gold, raw in samples))
    contains = sum((contains_answer(raw, gold) for gold, raw in samples))
    n = len(samples)
    f1_raw_mean = sum((f1(raw, gold) for gold, raw in samples)) / n
    f1_clean_mean = sum((f1(clean_generation(raw), gold) for gold, raw in samples)) / n
    print(f'{label}: EM_raw={em_raw}/{n} EM_clean={em_clean}/{n} F1_raw={f1_raw_mean:.3f} F1_clean={f1_clean_mean:.3f} contains={contains}/{n}')
    ok = em_raw == expected_em_raw and em_clean == expected_em_clean and (contains == expected_contains)
    print(f"  {('PASS' if ok else 'MISMATCH vs expected — check patch against phase1_fixes_task.md')}")
    return ok
r1 = check(qwen_samples, 'Qwen3-8b', expected_em_raw=2, expected_em_clean=4, expected_contains=19)
r2 = check(ministral_samples, 'Ministral-3-8b', expected_em_raw=7, expected_em_clean=7, expected_contains=19)
if r1 and r2:
    print('\nAll checks passed — cleanup function matches validated behavior.')
else:
    print('\nSomething diverged from the validated baseline — do not assume the new numbers are correct.')
