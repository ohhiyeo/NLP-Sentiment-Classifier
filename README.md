This is a unigram logistic regression classifier for predicting the sentiment (negative or positive) of a movie review.

How to run:  
1. The code at line 9-11 are the negation words, negation enders and sentence enders:  
```ruby
negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])
```
Feel free to modify it.

2. Download Sentiment.py, train.txt, test.txt to the same file
3. Go to terminal -> go to the directory where those 3 files reside.
4. Run command:
```
$ python3 Sentiment.py train.txt test.txt
```

The result shows:  
1. (Precision, Recall, F-Measure) of the test.txt  
2. Show 5 top features and its weight  
