"""
Tokenization demo: sentence and word tokenizers with NLTK.
"""
import nltk
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer

from common import SAMPLE_CORPUS


def ensure_resources():
    """Download NLTK data if needed (run once)."""
    nltk.download("punkt_tab", quiet=True)


def run_demo(corpus: str = SAMPLE_CORPUS) -> None:
    ensure_resources()

    print("=== Sentence tokenization ===")
    sentences = sent_tokenize(corpus)
    print(f"Type: {type(sentences).__name__}")
    print(sentences)

    print("\n=== Word tokenization (wordpunct) — separates trailing punctuation ===")
    for sentence in sentences:
        tokens = wordpunct_tokenize(sentence)
        print(tokens)

    print("\n=== Word tokenization (Treebank) — keeps full stop attached, e.g. 'end.' ===")
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(corpus)
    print(tokens)


if __name__ == "__main__":
    run_demo()
