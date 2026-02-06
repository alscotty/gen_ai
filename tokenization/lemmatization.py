"""
Lemmatization demo: WordNet lemmatizer with NLTK.
Lemmatizer reduces words to a dictionary form (valid word); requires POS for best results.
"""
import nltk
from nltk.stem import WordNetLemmatizer

from common import WORDS


def ensure_resources() -> None:
    """Download WordNet (and OMW) if needed (run once)."""
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def run_demo(word_list: list[str] | None = None) -> None:
    ensure_resources()
    words = word_list or WORDS
    lemmatizer = WordNetLemmatizer()

    print("Lemmatize as verb (pos='v'):")
    for w in words:
        print(f"  {w} -> {lemmatizer.lemmatize(w, pos='v')}")

    print("\nLemmatize as noun (pos='n', default):")
    for w in words:
        print(f"  {w} -> {lemmatizer.lemmatize(w, pos='n')}")

    # POS: n=noun, v=verb, a=adjective, r=adverb
    print("\nExample with explicit POS: better (adj) -> good:")
    print(f"  better -> {lemmatizer.lemmatize('better', pos='a')}")


if __name__ == "__main__":
    run_demo()
