import sentiment_mod as s

test = open("test review.txt", "r").read()

print(s.sentiment(test))