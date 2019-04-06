import warnings
warnings.filterwarnings("ignore")
from os.path import join
from nltk.sentiment.util import mark_negation
from features.sentiment.senti_strength.senti_strength import senti_strength_class
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from features.sentiment.afinn import Afinn
from features.sentiment.hu_liu.hu_liu import hu_liu_class
from features.sentiment.senti_wordNet.sentiment import SentimentAnalysis
from features.sentiment.effect_wordNet.effect_wordNet import effect_wordNet_class
from features.sentiment.subjectivity_clues.subjectivity_clues import subjectivity_clues_class
from features.sentiment.sentic_net.sentic_net import sentic_net_class
from features.sentiment.emo_lex.emo_lex import emo_lex_class

class Sentiment:

    def __init__(self, path=''):
        self.Vader = SentimentIntensityAnalyzer()
        self.senti_Strength = senti_strength_class(path=join(path, 'senti_strength/'))
        self.afn = Afinn()
        self.huliu = hu_liu_class(path=join(path, 'hu_liu/'))
        self.senti_wordNet = SentimentAnalysis(filename= join(path,'senti_wordNet/SentiWordNet_3.0.txt'), weighting='geometric')
        self.effect_WN = effect_wordNet_class(path=join(path, 'effect_wordNet/'))
        self.sentic_net = sentic_net_class(path=join(path, 'sentic_net/'))
        self.subj_cue_senti = subjectivity_clues_class(path=join(path, 'subjectivity_clues/'))
        self.emo_lex_senti = emo_lex_class(path=join(path, 'emo_lex/'))

    def Vader_API(self, sentence):
        return list(self.Vader.polarity_scores(sentence).values())

    def one_vector_senti(self, sentence):
        sentence = ' '.join(mark_negation(sentence.split()))
        sentence = str(sentence).lower()
        global_vec = []
        global_vec.extend(self.Vader_API(sentence))
        global_vec.extend([self.senti_Strength.score(sentence)])
        global_vec.extend([self.afn.score(sentence)])
        global_vec.extend(self.huliu.score(sentence))
        global_vec.extend([self.senti_wordNet.score(sentence)])
        global_vec.extend(self.effect_WN.score(sentence))
        global_vec.extend(self.sentic_net.score(sentence))
        global_vec.extend(self.subj_cue_senti.score(sentence))
        global_vec.extend(self.emo_lex_senti.score(sentence))
        return global_vec

if __name__ == '__main__':
    snt = Sentiment()
    s = ["I don't like the movie, it's bad I hate it", "I love you sweety kiss", 'I was so sad, he called me the bitch, he was killed']
    print(snt.one_vector_senti(s[0]))
    # print(snt.Vader_API(s[0]))
    # print([snt.senti_Strength.score(s[0])])
    # print([snt.afn.score(s[0])])
    # print(snt.huliu.score(s[0]))
    # print([snt.senti_wordNet.score(s[0])])
    # print(snt.effect_WN.score(s[0]))
    # print(snt.sentic_net.score(s[0]))
    # print(snt.subj_cue_senti.score(s[0]))
    # print(snt.emo_lex_senti.score(s[0]))
