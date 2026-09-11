# Tokenizer Efficiency Across Languages

How many tokens does it cost to say the same thing in different languages? I'm trilingual (Russian, Mongolian, English) and kept noticing that AI tools — translation, speech, pronunciation — consistently struggled more with Mongolian than with my other languages. This project tests one concrete, measurable reason why: **tokenizers don't encode all languages equally efficiently**, and that inefficiency eats directly into a model's usable context window and inference cost before it ever gets to "understanding" anything.

## Method

- **Data**: [FLORES-200](https://github.com/facebookresearch/flores), Meta's parallel evaluation corpus — the same set of sentences, professionally translated into 200 languages. Using parallel data (not independently written sentences) means differences in token count reflect the tokenizer, not differences in what was said.
- **Languages tested**: English (baseline), Russian, Mongolian (Halh, Cyrillic script), Japanese, Czech.
- **Tokenizers tested**: GPT-4 (`cl100k_base`), GPT-4o (`o200k_base`), Llama-3, mBERT, Mistral-7B, NLLB-200 (a tokenizer purpose-built for 200 languages).
- **Metric**: total token count per language, normalized against English (e.g. a ratio of 2.0 means that language took twice as many tokens as English for the same content).

## Findings

Mongolian was the least efficiently tokenized language across **every tokenizer tested — 6 for 6** — ranging from **1.39x** more tokens (NLLB-200) up to **3.74x** more tokens (GPT-4's `cl100k_base`) than English for identical content.

| Language  | GPT-4 (cl100k) | GPT-4o (o200k) | Llama-3 | mBERT | Mistral | NLLB-200 |
|-----------|---------------:|----------------:|--------:|------:|--------:|---------:|
| Czech     | 1.99x | 1.54x | 1.45x | 1.38x | 1.74x | 1.23x |
| Japanese  | 2.27x | 1.68x | 1.52x | 1.56x | 2.11x | 1.06x |
| Mongolian | **3.74x** | **1.91x** | **3.15x** | **1.96x** | **2.88x** | **1.39x** |
| Russian   | 2.49x | 1.46x | 1.66x | 1.46x | 1.84x | 1.36x |

![Tokenizer efficiency chart](tokenizer_efficiency_chart.png)

The clearest pattern is by tokenizer *purpose*, not release date or model family. NLLB-200 — trained specifically for multilingual coverage across 200 languages — is the best performer on every single language tested. General-purpose tokenizers, including Llama-3 (a comparatively recent, widely-used open model), show 2-3x worse Mongolian efficiency than NLLB, landing much closer to GPT-4 and Mistral than to NLLB. That consistency across otherwise very different models suggests the gap isn't a quirk of any one lab's training data — it's what happens by default when multilingual coverage isn't an explicit design goal.

This isn't just a translation-quality footnote. More tokens per unit of meaning means less effective context window and higher inference cost for speakers of under-served languages — and NLLB's numbers show this gap is a design choice, not an unavoidable limit of the technology.

## Next steps

- Scale from 200 sentences to the full FLORES-200 dev set (~1000 sentences) to confirm the ratios hold.
- Test whether the gap is about the *language* or the *script*, by comparing Mongolian in Cyrillic (`khk_Cyrl`) against traditional Mongolian script (`mon_Mong`).
- Translate the token ratio into a real cost figure (e.g. $/word) using current API pricing.

## Running it yourself

```bash
python3 -m venv tokenizer-env
source tokenizer-env/bin/activate
pip install -r requirements.txt
hf auth login   # free account at huggingface.co; needed for NLLB, mBERT, gated models
jupyter notebook
```

Open `Tokenizer Efficiency Across Languages.ipynb` and run all cells top to bottom. First run will download tokenizer files and the FLORES dataset (a few hundred MB); after that it's cached locally.

Note: `facebook/flores` and gated models (Llama-3, Mistral) require clicking "Agree and access repository" on their Hugging Face pages before your token will work.
