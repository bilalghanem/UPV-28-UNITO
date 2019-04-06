"""
Class to score sentiment of text.

Use domain-independent method of dictionary lookup of sentiment words,
handling negations and multiword expressions. Based on SentiWordNet 3.0.

"""

import nltk
import re


class SentimentAnalysis(object):
    """Class to get sentiment score based on analyzer."""

    def __init__(self, filename='SentiWordNet.txt', weighting='geometric'):
        """Initialize with filename and choice of weighting."""
        if weighting not in ('geometric', 'harmonic', 'average'):
            raise ValueError(
                'Allowed weighting options are geometric, harmonic, average')
        # parse file and build sentiwordnet dicts
        self.swn_pos = {'a': {}, 'v': {}, 'r': {}, 'n': {}}
        self.swn_all = {}
        self.build_swn(filename, weighting)

    def average(self, score_list):
        """Get arithmetic average of scores."""
        if(score_list):
            return sum(score_list) / float(len(score_list))
        else:
            return 0

    def geometric_weighted(self, score_list):
        """"Get geometric weighted sum of scores."""
        weighted_sum = 0
        num = 1
        for el in score_list:
            weighted_sum += (el * (1 / float(2**num)))
            num += 1
        return weighted_sum

    # another possible weighting instead of average
    def harmonic_weighted(self, score_list):
        """Get harmonic weighted sum of scores."""
        weighted_sum = 0
        num = 2
        for el in score_list:
            weighted_sum += (el * (1 / float(num)))
            num += 1
        return weighted_sum

    def build_swn(self, filename, weighting):
        """Build class's lookup based on SentiWordNet 3.0."""
        records = [line.split('\t') for line in open(filename)]
        for rec in records:
            # has many words in 1 entry
            words = rec[4].split()
            pos = rec[0]
            for word_num in words:
                word = word_num.split('#')[0]
                sense_num = int(word_num.split('#')[1])

                # build a dictionary key'ed by sense number
                if word not in self.swn_pos[pos]:
                    self.swn_pos[pos][word] = {}
                self.swn_pos[pos][word][sense_num] = float(
                    rec[2]) - float(rec[3])
                if word not in self.swn_all:
                    self.swn_all[word] = {}
                self.swn_all[word][sense_num] = float(rec[2]) - float(rec[3])

        # convert innermost dicts to ordered lists of scores
        for pos in self.swn_pos.keys():
            for word in self.swn_pos[pos].keys():
                newlist = [self.swn_pos[pos][word][k] for k in sorted(
                    self.swn_pos[pos][word].keys())]
                if weighting == 'average':
                    self.swn_pos[pos][word] = self.average(newlist)
                if weighting == 'geometric':
                    self.swn_pos[pos][word] = self.geometric_weighted(newlist)
                if weighting == 'harmonic':
                    self.swn_pos[pos][word] = self.harmonic_weighted(newlist)

        for word in self.swn_all.keys():
            newlist = [self.swn_all[word][k] for k in sorted(
                self.swn_all[word].keys())]
            if weighting == 'average':
                self.swn_all[word] = self.average(newlist)
            if weighting == 'geometric':
                self.swn_all[word] = self.geometric_weighted(newlist)
            if weighting == 'harmonic':
                self.swn_all[word] = self.harmonic_weighted(newlist)

    def pos_short(self, pos):
        """Convert NLTK POS tags to SWN's POS tags."""
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'a'

    def score_word(self, word, pos):
        """Get sentiment score of word based on SWN and part of speech."""
        try:
            return self.swn_pos[pos][word]
        except KeyError:
            try:
                return self.swn_all[word]
            except KeyError:
                return 0

    def score(self, sentence):
        """Sentiment score a sentence."""
        # init sentiwordnet lookup/scoring tools
        impt = set(['NNS', 'NN', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
                    'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
                    'VBP', 'VBZ', 'unknown'])
        non_base = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
        negations = set(['not', 'n\'t', 'less', 'no', 'never',
                         'nothing', 'nowhere', 'hardly', 'barely',
                         'scarcely', 'nobody', 'none'])
        stopwords = nltk.corpus.stopwords.words('english')
        wnl = nltk.WordNetLemmatizer()

        scores = []
        tokens = nltk.tokenize.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        index = 0
        for el in tagged:

            pos = el[1]
            try:
                word = re.match('(\w+)', el[0]).group(0).lower()
                start = index - 5
                if start < 0:
                    start = 0
                neighborhood = tokens[start:index]

                # look for trailing multiword expressions
                word_minus_one = tokens[index-1:index+1]
                word_minus_two = tokens[index-2:index+1]

                # if multiword expression, fold to one expression
                if(self.is_multiword(word_minus_two)):
                    if len(scores) > 1:
                        scores.pop()
                        scores.pop()
                    if len(neighborhood) > 1:
                        neighborhood.pop()
                        neighborhood.pop()
                    word = '_'.join(word_minus_two)
                    pos = 'unknown'

                elif(self.is_multiword(word_minus_one)):
                    if len(scores) > 0:
                        scores.pop()
                    if len(neighborhood) > 0:
                        neighborhood.pop()
                    word = '_'.join(word_minus_one)
                    pos = 'unknown'

                # perform lookup
                if (pos in impt) and (word not in stopwords):
                    if pos in non_base:
                        word = wnl.lemmatize(word, self.pos_short(pos))
                    score = self.score_word(word, self.pos_short(pos))
                    if len(negations.intersection(set(neighborhood))) > 0:
                        score = -score
                    scores.append(score)

            except AttributeError:
                pass

            index += 1

        if len(scores) > 0:
            return sum(scores) / float(len(scores))
        else:
            return 0

    def is_multiword(self, words):
        """Test if a group of words is a multiword expression."""
        joined = '_'.join(words)
        return joined in self.swn_all

