import numpy as np
import math
import glob
import re
from nltk.corpus import stopwords


class wordCounter(object):
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''
    def __init__(self, word, pos, neg, p):
        self.word = word
        self.pos  = pos
        self.neg  = neg
        self.p    = p


class naiveBayes(object):
    '''
    Naive bayes class
    Train model and classify new emails
    '''
    def _extractWords(self, filecontent):
        '''
        Word extractor from filecontent
        :param filecontent: filecontent as a string
        :return: list of words found in the file
        '''
        txt = filecontent.split(" ")
        txtClean = [(re.sub(r'[^a-zA-Z]+','', i).lower()) for i in txt]
        words = [i for i in txtClean if i.isalpha() ]
        return words


    def train(self, msgDirectory, fileFormat='*.txt'):
        '''
        :param msgDirectory: Directory to email files that should be used to train the model
        :return: model dictionary and model prior
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        # TODO: Train the naive bayes classifier
        # TODO: Hint - store the dictionary as a list of 'wordCounter' objects
        self.logPrior = ()
        self.dictionary = []
        for file in files:
            file_opened = open(file, 'r')
            string = file_opened.read()
            string = self._extractWords(string)
            stop_words = set(stopwords.words('english'))
            string = [w for w in string if not w in stop_words]
            for item in string:
                index = -1
                temp_word_counter = wordCounter(word=item, pos=0, neg=0, p=0)
                for i in range(len(self.dictionary)):
                    if self.dictionary[i].word == item:
                        index = i
                        temp_word_counter = self.dictionary[i]

                if file.startswith('s'):
                    temp_word_counter.pos += 1
                else:
                    temp_word_counter.neg += 1

                temp_word_counter.p += 1
                if index > -1:
                    self.dictionary[index] = temp_word_counter
                else:
                    self.dictionary.append(temp_word_counter)

        return (self.dictionary, self.logPrior)


    def classify(self, message, nFeatures):
        '''
        :param message: Input email message as a string
        :param nFeatures: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''

        txt = np.array(self._extractWords(message))
        # TODO: Implement classification function


    def classifyAndEvaluateAllInFolder(self, msgDirectory, nFeatures, fileFormat='*.txt'):
        '''
        :param msgDirectory: Directory to email files that should be classified
        :param nFeatures: Number of features to be used from the trained dictionary
        :return: Classification accuracy
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        corr = 0    # Number of correctly classified messages
        ncorr = 0   # Number of falsely classified messages
        # TODO: Classify each email found in the given directory and figure out if they are correctly or falsely classified
        # TODO: Hint - look at the filenames to figure out the ground truth label
        return corr/(corr+ncorr)


    def printMostPopularSpamWords(self, num):
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary


    def printMostPopularHamWords(self, num):
        print("{} most popular HAM words:".format(num))
        # TODO: print the 'num' most used HAM words from the dictionary


    def printMostindicativeSpamWords(self, num):
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary


    def printMostindicativeHamWords(self, num):
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
