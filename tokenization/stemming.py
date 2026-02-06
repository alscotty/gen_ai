"""
Stemming demo: Porter, Regexp, and Snowball stemmers with NLTK.
Stemmers reduce words to a root form (may not be a valid word).
"""
from nltk.stem import PorterStemmer, RegexpStemmer, SnowballStemmer

from common import EDGE_WORDS, WORDS


def run_porter_demo(words: list[str]) -> None:
    """Porter stemmer: rule-based, English, no language arg."""
    stemmer = PorterStemmer()
    print("Porter:")
    for w in words:
        print(f"  {w} -> {stemmer.stem(w)}")
    print(f"  congratulations -> {stemmer.stem('congratulations')}")


def run_regexp_demo(words: list[str]) -> None:
    """Regexp stemmer: strip suffixes matching pattern (e.g. ing$, s$, ed$)."""
    # min=4 avoids over-stripping short words
    stemmer = RegexpStemmer("ing$|s$|ed$|'mer$", min=4)
    print("Regexp (ing$|s$|ed$|'mer$, min=4):")
    for w in words:
        print(f"  {w} -> {stemmer.stem(w)}")
    print(f"  congratulations -> {stemmer.stem('congratulations')}")


def run_snowball_demo(words: list[str]) -> None:
    """Snowball stemmer: multilingual, default English."""
    stemmer = SnowballStemmer("english")
    print("Snowball (english):")
    for w in words:
        print(f"  {w} -> {stemmer.stem(w)}")
    print(f"  congratulations -> {stemmer.stem('congratulations')}")


def run_edge_comparison() -> None:
    """Compare Porter vs Snowball on edge words (e.g. fairly, sportingly)."""
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    print("Edge words (Porter vs Snowball):")
    for w in EDGE_WORDS:
        print(f"  {w}: Porter={porter.stem(w)}, Snowball={snowball.stem(w)}")


if __name__ == "__main__":
    run_porter_demo(WORDS)
    print()
    run_regexp_demo(WORDS)
    print()
    run_snowball_demo(WORDS)
    print()
    run_edge_comparison()
