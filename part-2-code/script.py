from transformers import T5TokenizerFast
import numpy as np

tok = T5TokenizerFast.from_pretrained("t5-small")

def read_lines(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

def mean_len_tokenized(lines):
    return float(np.mean([len(tok.encode(s, add_special_tokens=True)) for s in lines]))

def vocab_size_used(lines):
    used = set()
    for s in lines:
        used.update(tok.encode(s, add_special_tokens=False))
    return len(used)

def report(split):
    nl = read_lines(f"data/{split}.nl")
    sql = read_lines(f"data/{split}.sql")
    print(f"\n=== {split.upper()} (BEFORE) ===")
    print("Num examples:", len(nl))
    print("Mean NL length (tokens):", mean_len_tokenized(nl))
    print("Mean SQL length (tokens):", mean_len_tokenized(sql))
    print("Vocab size used (NL):", vocab_size_used(nl))
    print("Vocab size used (SQL):", vocab_size_used(sql))

    # AFTER preprocessing: add prompt prefix and trunc to 128 for inputs/labels
    prefixed = [f"translate to SQL: {s}" for s in nl]
    print(f"\n=== {split.upper()} (AFTER preprocess) ===")
    print("Model name: t5-small")
    print("Mean NL length (tokens) with prefix+trunc@128:",
          float(np.mean([len(tok.encode(s, add_special_tokens=True, max_length=128, truncation=True)) for s in prefixed])))
    print("Mean SQL length (tokens) with trunc@128:",
          float(np.mean([len(tok.encode(s, add_special_tokens=True, max_length=128, truncation=True)) for s in sql])))
    print("Vocab size used (NL, after prefix):", vocab_size_used(prefixed))
    print("Vocab size used (SQL, after):", vocab_size_used(sql))

if __name__ == "__main__":
    report("train")
    report("dev")