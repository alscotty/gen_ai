import nltk
nltk.download('punkt_tab')

corpus = """
Hello, my name is John Doe.
I am a software engineer!
I live in New York.
"""

from nltk.tokenize import sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer

sentences = sent_tokenize(corpus)
print("type(sentences): ", type(sentences))
print(sentences)

# good for most use cases, separates full stop punctuation at end of sentence
for sentence in sentences:
    print(wordpunct_tokenize(sentence))

# this keeps full stop attached to work, ex. "end." instead of "end","."
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(corpus)

print(tokenizer)