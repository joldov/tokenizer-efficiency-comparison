import tiktoken
from transformers import AutoTokenizer
from datasets import load_dataset

LANG_CODES = {
    "english": "eng_Latn",
    "russian": "rus_Cyrl",
    "mongolian": "khk_Cyrl",   # Halh Mongolian, Cyrillic script
    "japanese": "jpn_Jpan",
    "czech": "ces_Latn",
}

N_SENTENCES = 200  # start small, scale up once the pipeline works

def load_flores_sentences():
    sentences = {}
    for lang_name, code in LANG_CODES.items():
        ds = load_dataset("facebook/flores", code, split="dev", trust_remote_code=True)
        sentences[lang_name] = [row["sentence"] for row in ds][:N_SENTENCES]
    return sentences

def get_tokenizers():
    tokenizers = {}

    tokenizers["gpt-4o (o200k_base)"] = ("tiktoken", tiktoken.get_encoding("o200k_base"))
    tokenizers["gpt-4 (cl100k_base)"] = ("tiktoken", tiktoken.get_encoding("cl100k_base"))

    hf_models = {
        "llama-3": "meta-llama/Meta-Llama-3-8B",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "nllb-200": "facebook/nllb-200-distilled-600M",  # tokenizer built for 200 languages
        "mbert": "bert-base-multilingual-cased",
    }
    for name, repo in hf_models.items():
        try:
            tokenizers[name] = ("hf", AutoTokenizer.from_pretrained(repo))
        except Exception as e:
            print(f"Skipping {name}: {e}")

    return tokenizers


def count_tokens(tokenizer_type, tokenizer, text):
    if tokenizer_type == "tiktoken":
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.encode(text))
        
def main():
    sentences = load_flores_sentences()
    tokenizers = get_tokenizers()

    results = {tok_name: {} for tok_name in tokenizers}

    for tok_name, (tok_type, tok) in tokenizers.items():
        for lang, sents in sentences.items():
            total_tokens = sum(count_tokens(tok_type, tok, s) for s in sents)
            results[tok_name][lang] = total_tokens

    print(f"{'Tokenizer':<22} {'Lang':<10} {'Tokens':>8} {'Ratio vs EN':>12}")
    print("-" * 56)
    for tok_name, lang_counts in results.items():
        en_count = lang_counts["english"]
        for lang, count in lang_counts.items():
            ratio = count / en_count
            print(f"{tok_name:<22} {lang:<10} {count:>8} {ratio:>11.2f}x")
        print()

    return results


if __name__ == "__main__":
    main()
