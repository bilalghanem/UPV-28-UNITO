import sys


class senti_strength_class:
    def __init__(self, path=''):
        self.sentimentData = path + 'wordwithStrength.txt'
        self.twitterData = 'input.txt'
        self.sentiment = self.sentiment_dict(self.sentimentData)

        #var will take the user input
        # var = input("Enter something: ")

    def tweet_dict(self, twitterData):
        ''' (file) -> list of dictionaries
        This method should take your output.txt
        file and create a list of dictionaries.
        '''
        twitter_list_dict = []
        #storing the user given sentance to the dictionary for the classification
        # twitter_list_dict.extend([var])
        twitter_list_dict = ["it is good not bad","good movie i love it", "hate this, really hate this movie", "you can use this if you want to work with direct list with data"]
        return twitter_list_dict

    def sentiment_dict(self, sentimentData):
        ''' (file) -> dictionary
        This method should take your sentiment file
        and create a dictionary in the form {word: value}
        '''

        afinnfile = open(sentimentData)
        scores = {} # initialize an empty dictionary
        for line in afinnfile:
            term, score = line.split("\t") # The file is tab-delimited. "\t" means "tab character"
            scores[term] = float(score) # Convert the score to an integer.

        return scores # Print every (term, score) pair in the dictionary

    def score(self, tweets):
        # tweets = tweet_dict(twitterData)
        '''Create a method below that loops through each tweet in your
        twees_list. For each individual tweet it should add up you sentiment
        score, based on the sent_dict.
        '''
        tweet_word = tweets.split()
        #sent_score is a variable which will take care of word strength / word weightage
        sent_score = 0
        for word in tweet_word:
            word = word.rstrip('?:!.,;"!@')
            word = word.replace("\n", "")
            if word in self.sentiment.keys():
                sent_score = sent_score + float(self.sentiment[word])
            else:
                sent_score = sent_score
        if float(sent_score) > 0:
            # print(tweets[index])
            if float(sent_score) > 0.7:
                # print('Highly Positive Sentiment')
                tweets = 1.0
            else:
                # print('Positive Sentiment')
                tweets = 0.75

        if float(sent_score) < 0:
            # print(tweets[index])
            if float(sent_score) < -0.7:
                # print('Highly Negative Sentiment')
                tweets = 0
            else:
                # print('Negative Sentiment')
                tweets = 0.25

        if float(sent_score) == 0:
            # print('Neutral Sentiment')
            tweets = 0.5
        return tweets
   
if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    SS = senti_strength_class()
    print(SS.main_sentist(s[0]))
