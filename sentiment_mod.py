import nltk
import random
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


short_pos = open("Dataset/positive.txt", "r").read()
short_neg = open("Dataset/negative.txt", "r").read()

save_documents = open("Pickles/documents.pickle","rb")
documents = pickle.load(save_documents)
save_documents.close()

save_word_features = open("Pickles/word_features.pickle","rb")
word_features = pickle.load(save_word_features)
save_word_features.close()

save_featuresets = open("Pickles/featuresets.pickle", "rb")
featuresets = pickle.load(save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

save_classifier = open("Pickles/NB.pickle","rb")
classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/MNB.pickle","rb")
MNB_classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/Bernoulli_classifier.pickle","rb")
Bernoulli_classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("Pickles/NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  Bernoulli_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)



def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

