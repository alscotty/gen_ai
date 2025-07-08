# import nltk
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from stemming import words

for word in words:
    print(f"{word} -> {lemmatizer.lemmatize(word, pos="v")}")

# pos can be "n" for noun, "v" for verb, "a" for adjective, "r" for adverb
# default is "n"
# can also provide a part of speech tag, e.g. "v" for verb, "n" for noun, "a" for adjective, "r" for adverb
# can also provide a list of part of speech tags, e.g. ["v", "n", "a", "r"]
# can also provide a function that returns a part of speech tag, e.g. lambda word: "v" if word.endswith("ing") else "n"