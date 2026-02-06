"""
Shared sample data for tokenization, stemming, and lemmatization demos.
"""

# Words used for stemming/lemmatization comparison
WORDS = [
    "eating", "eats", "eat", "ate", "eated",
    "programming", "programmer", "program",
]

# Edge cases: stemmers differ (e.g. fairly -> fair vs fairli)
EDGE_WORDS = ["fairly", "sportingly"]

# Sample corpus for sentence/word tokenization
SAMPLE_CORPUS = """
Hello, my name is John Doe.
I am a software engineer!
I live in New York.
""".strip()
