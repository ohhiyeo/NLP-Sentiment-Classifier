import re
import sys
import csv
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    # t = open(corpus_path, encoding='utf-8', errors='ignore')
    # paragraph = t.read().split('\n')
    # for p in paragraph:
    #     review, c = p.split('\t')
    #     r = review.split()
    #     finalList.append((r,c))
    finalList = []
    with open(corpus_path, encoding='utf-8', errors='ignore') as file:
        file_reader = csv.reader(file, delimiter='\t')
        for row in file_reader:
            r = []
            r = row[0].split()
            finalList.append((r,int(row[1])))

    return finalList

# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words:
        return True
    elif word.endswith("n't"):
        return True
    else:
        return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    #snippet = one review
    ntagList = snippet
    result = nltk.pos_tag(snippet)
    taggable = False
    for index, obj in enumerate(result):
        if obj[0] in sentence_enders or obj[0] in negation_enders or obj[1] == "JJR" or obj[1] == "RBR":
            taggable = False
            continue

        if(is_negation(obj[0])):
            if obj[0] == "not" and index < (len(result)-1) and result[index+1][0] == "only":
                taggable = False
                continue
            else:
                taggable = True
        elif taggable:
            ntagList[index] = "NOT_"+ ntagList[index]
            taggable = False
       
        

    return ntagList


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    vocabIndex_dict = {}
    idxCount = 0
    for obj in corpus:
        for token in obj[0]:
            if token not in vocabIndex_dict:
                vocabIndex_dict.update({token:idxCount})
                idxCount+=1
    return vocabIndex_dict
    
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    vectorNP = np.zeros(len(feature_dict))
    for word in snippet:
        if word not in feature_dict:
            continue
        idx = feature_dict.get(word)
        vectorNP[idx]+=1


    return vectorNP


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    n = len(corpus)
    d = len(feature_dict)
    vectorCorpX = np.empty([n,d],dtype=float)
    vectorCorpY = np.empty(n,dtype=float)
    for index, obj in enumerate(corpus):
        vectorCorpX[index] = vectorize_snippet(obj[0], feature_dict)
        vectorCorpY[index] = obj[1]
    
    return (vectorCorpX,vectorCorpY)


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    
    for column in range(X.shape[1]):
        maxValue = np.max(X[:,column])
        minValue = np.min(X[:,column])
        
        if maxValue > 0 and minValue - maxValue != 0:
            X[:,column] = (X[:,column] - minValue)/(maxValue - minValue)

# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    review_label_Tup = load_corpus(corpus_path)
    newReview = []
    [newReview.append(list(l)) for l in review_label_Tup]
    for index,obj in enumerate(newReview):
        newReview[index][0] = tag_negation(obj[0])
       
    review_label_Tup = []
    [review_label_Tup.append(tuple(l)) for l in newReview]
    feature_dict = get_feature_dictionary(review_label_Tup)
    vectored = vectorize_corpus(review_label_Tup,feature_dict)
    normalize(vectored[0])
    log_model = LogisticRegression()
    model = log_model.fit(vectored[0],vectored[1])
    return (model, feature_dict)
    


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp = 0
    fp = 0
    fn = 0
    for index in range(len(Y_pred)):
        if Y_test[index] == 1:
            if Y_pred[index]==1:
                tp+=1  
            else:
                fn+=1
        else:
            if Y_pred[index]==1:
                fp+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F_measure = ((precision*recall)/(precision+recall))*2
    return (precision, recall, F_measure)


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    test_label_Tup = load_corpus(corpus_path)
    newReview = []
    [newReview.append(list(l)) for l in test_label_Tup]
    for index,obj in enumerate(newReview):
        newReview[index][0] = tag_negation(obj[0])
    review_label_Tup = []
    [review_label_Tup.append(tuple(l)) for l in newReview]
    vectored = vectorize_corpus(review_label_Tup,feature_dict)
    normalize(vectored[0])
    Y_pred = model.predict(vectored[0])
    evaluation = evaluate_predictions(Y_pred, vectored[1])
    return evaluation


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    logregCoefficient = logreg_model.coef_
    logregList = list()
    lr = logregCoefficient[0]
    for index, coefficient in enumerate(lr):
        logregList.append(tuple((index,coefficient)))
    logregList.sort(key=lambda x:abs(x[1]),reverse=True)
    keyList = list(feature_dict.keys())
    valueList = list(feature_dict.values())
    weighted_feature = []
    for index in range(k):
        k = keyList[valueList.index(logregList[index][0])]
        weighted_feature.append((k,logregList[index][1]))

    return weighted_feature

    


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict,5)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
