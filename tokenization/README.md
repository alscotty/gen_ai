# Tokenization

Intro to tokenization, stemming, and lemmatization with NLTK. Scripts split text into sentences/words and reduce word forms for downstream NLP.

## Files

- **common.py** – Shared sample data: `WORDS`, `EDGE_WORDS`, `SAMPLE_CORPUS` used by the demos.
- **example.py** – Sentence and word tokenization:
  - `sent_tokenize` for sentences
  - `wordpunct_tokenize` (punctuation separated)
  - `TreebankWordTokenizer` (punctuation attached, e.g. `end.`)
- **stemming.py** – Stemming comparison:
  - **Porter**: rule-based English
  - **Regexp**: pattern-based (e.g. `ing$`, `s$`, `ed$`)
  - **Snowball**: multilingual (default English), edge-case comparison with Porter
- **lemmatization.py** – WordNet lemmatization with POS:
  - `pos='v'` (verb) vs `pos='n'` (noun); optional adjective example

## Concepts

- **Tokenization**: Split text into sentences or words; choice of tokenizer affects punctuation handling.
- **Stemming**: Reduce words to a root (may not be a valid word); fast, no dictionary.
- **Lemmatization**: Reduce to dictionary form (valid word); usually needs part-of-speech for best results.

## Setup

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip install -r requirements.txt
```

First run will download NLTK data (e.g. `punkt_tab`, `wordnet`, `omw-1.4`) when needed.

## Run

```bash
python example.py
python stemming.py
python lemmatization.py
```

## Requirements

- **requirements.txt**: `nltk`
