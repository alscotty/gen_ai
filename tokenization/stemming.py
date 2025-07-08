words = ["eating", "eats", "eat", "ate", "eated", "programming", "programmer", "program"]
edge_words = ["fairly", "sportingly"]

from nltk.stem import PorterStemmer, RegexpStemmer, SnowballStemmer

porter = PorterStemmer()

# for word in words:
    # print(f"{word} -> {porter.stem(word)}")

# print(porter.stem("congratulations")) # congratulations -> congratul

reg_exp = RegexpStemmer("ing$|s$|ed$|'mer$", min=4) # matches words ending in ing, s, ed, or 'mer

# for word in words:
#     print(f"{word} -> {reg_exp.stem(word)}")

# print(reg_exp.stem("congratulations")) # congratulations -> congratulutation

snowball = SnowballStemmer("english")
# can provide language as argument, default is english, can also use "danish", "finnish", "french", "german", "hungarian", "italian", "norwegian", "portuguese", "romanian", "russian", "spanish", "swedish", "turkish" etc.

if __name__ == "__main__":
    for word in words:
        print(f"{word} -> {snowball.stem(word)}")
    print(snowball.stem("congratulations")) # congratulations -> congratul
    print("MORE SNOWBALL STEMMING")
    print(snowball.stem("fairly")) # fairly -> fair
    print(snowball.stem("sportingly")) # sportingly -> sport
    print("VS PORTER STEMMING")
    print(porter.stem("fairly")) # fairly -> fairli
    print(porter.stem("sportingly")) # sportingly -> sportli
