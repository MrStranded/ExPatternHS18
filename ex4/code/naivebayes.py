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
        self.dictionary = list()
        self.spam_list = list()
        self.ham_list = list()
        num_spam_mails = 0
        num_ham_mails = 0

        stop_words = set(stopwords.words('english'))

        for file in files:
            file_content = open(file)
            words = self._extractWords(file_content.read())
            words_cleaned = [value for value in words if value not in stop_words]


            if 'spmsga' in file:
                # Spam mail
                for word in words_cleaned:
                    not_found = True
                    for counter in self.dictionary:
                        if counter.word == word:
                            counter.neg += 1
                            not_found = False

                    if not_found:
                        word_counter = wordCounter(word,0,1,0)
                        self.dictionary.append(word_counter)
                        self.spam_list.append(word_counter)
                num_spam_mails += 1
            else:
                # ham mail
                for word in words_cleaned:
                    not_found = True
                    for counter in self.dictionary:
                        if counter.word == word:
                            counter.pos += 1
                            not_found = False

                    if not_found:
                        word_counter = wordCounter(word,1,0,0)
                        self.dictionary.append(word_counter)
                        self.ham_list.append(word_counter)

                num_ham_mails += 1

        # Update the relative frequency of a word w in the training set
        num_spam_words = 0
        num_ham_words = 0
        for word_counter in self.dictionary:
            num_spam_words += word_counter.neg
            num_ham_words += word_counter.pos

        for word_counter in self.dictionary:
            posterior_ham = (word_counter.pos+1)/num_ham_words
            posterior_spam = (word_counter.neg+1)/num_spam_words
            word_counter.p = np.log(posterior_spam/posterior_ham)

        prior_ham = num_ham_mails/len(files)
        prior_spam = num_spam_mails/len(files)
        self.logPrior = np.log(prior_spam/prior_ham)

        self.dictionary.sort(key=lambda x: np.abs(x.p), reverse=True)
        self.spam_list.sort(key=lambda x: np.abs(x.p), reverse=True)
        self.ham_list.sort(key=lambda x: np.abs(x.p), reverse=True)
        return (self.dictionary, self.logPrior)


    def classify(self, message, nFeatures):
        '''
        :param message: Input email message as a string
        :param nFeatures: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''

        txt = np.array(self._extractWords(message))
        # TODO: Implement classification function
        score = self.logPrior
        for word in txt:
            for word_counter in self.dictionary[:nFeatures]:
                if word_counter.word == word:
                    score += word_counter.p

        if score < 0:
            return False
        else:
            return True


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
        for file in files:
            spam = self.classify(open(file).read(),nFeatures)
            if 'spmsga' in file and spam:
                corr += 1
            elif 'spmsga' not in file and not spam:
                corr += 1
            else:
                ncorr += 1
        return corr/(corr+ncorr)


    def printMostPopularSpamWords(self, num):
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary
        self.spam_list.sort(key=lambda x: x.neg, reverse=True)
        for word in self.spam_list[:num]:
            print(word.word)
        self.spam_list.sort(key=lambda x: np.abs(x.p), reverse=True)



    def printMostPopularHamWords(self, num):
        print("{} most popular HAM words:".format(num))
        # TODO: print the 'num' most used HAM words from the dictionary
        self.ham_list.sort(key=lambda x: x.pos, reverse=True)
        for word in self.ham_list[:num]:
            print(word.word)
        self.ham_list.sort(key=lambda x: np.abs(x.p), reverse=True)


    def printMostindicativeSpamWords(self, num):
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary
        for word in self.spam_list[:num]:
            print(word.word)


    def printMostindicativeHamWords(self, num):
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
        for word in self.ham_list[:num]:
            print(word.word)
