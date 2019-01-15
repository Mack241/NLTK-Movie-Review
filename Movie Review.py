import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
   
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(20))
#print(all_words["great"])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#positive data example:
train_set = featuresets[:1900]
test_set = featuresets[1900:]

#negative data example:
train_set = featuresets[100:]
test_set = featuresets[:100]

#classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Accuracy Percentage: ", (nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(30)

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier Accuracy Percentage: ", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

BernoulliNB = SklearnClassifier(BernoulliNB())
BernoulliNB.train(train_set)
print("BernoulliNB Accuracy Percentage: ", (nltk.classify.accuracy(BernoulliNB, test_set))*100)

LogisticRegression = SklearnClassifier(LogisticRegression())
LogisticRegression.train(train_set)
print("LogisticRegression Accuracy Percentage: ", (nltk.classify.accuracy(LogisticRegression, test_set))*100)

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(train_set)
print("SGDClassifier Accuracy Percentage: ", (nltk.classify.accuracy(SGDClassifier, test_set))*100)

SVC = SklearnClassifier(SVC())
SVC.train(train_set)
print("SVC Accuracy Percentage: ", (nltk.classify.accuracy(SVC, test_set))*100)

LinearSVC = SklearnClassifier(LinearSVC())
LinearSVC.train(train_set)
print("LinearSVC Accuracy Percentage: ", (nltk.classify.accuracy(LinearSVC, test_set))*100)

NuSVC = SklearnClassifier(NuSVC())
NuSVC.train(train_set)
print("NuSVC Accuracy Percentage: ", (nltk.classify.accuracy(NuSVC, test_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB,
                                  LogisticRegression,
                                  SGDClassifier,
                                  LinearSVC,
                                  NuSVC)
print("voted_classifier accuracy percentage:",(nltk.classify.accuracy(voted_classifier, test_set))*100 )

#print("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence:", voted_classifier.confidence(test_set[0][0])*100)
#print("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence:", voted_classifier.confidence(test_set[1][0])*100)
#print("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence:", voted_classifier.confidence(test_set[2][0])*100)
#print("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence:", voted_classifier.confidence(test_set[3][0])*100)
#print("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence:", voted_classifier.confidence(test_set[4][0])*100)
#print("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence:", voted_classifier.confidence(test_set[5][0])*100)
