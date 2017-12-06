from nltk.corpus import wordnet

syns = wordnet.synsets("hello")

print(syns[1].lemmas()[1].name())
