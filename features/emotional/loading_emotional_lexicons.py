import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from os.path import join
from empath import Empath
from features.emotional.liwc_readDict import readDict
import re, operator
import numpy as np

class emotional_lexicons:

    def __init__(self, path):
        self.lexicons_path = path
        self.emotions = {'anger': 0,
                         'disgust': 0,
                         'joy': 0,
                         'sadness': 0,
                         'surprise': 0,
                         'fear': 0,
                         'trust': 0,
                         'positive_emotion': 0,
                         'negative_emotion': 0,
                         'anticipation': 0,
                         'ambiguous': 0,
                         'calmness': 0,
                         'despair': 0,
                         'hate': 0,
                         'hope': 0,
                         'like': 0,
                         'love': 0
                         }

        # emoSentiNet, Ekman: Anger	Disgust	Joy	Sad	Surprise Fear
        self.emoSentiNet = pd.read_excel(join(self.lexicons_path, 'emo_senti_net.xls'), sheet_name='EmoSenticNet', header=0)
        self.emoSentiNet.rename(columns={'Anger': 'anger', 'Disgust': 'disgust', 'Joy': 'joy', 'Sad': 'sadness', 'Surprise': 'surprise', 'Fear': 'fear'}, inplace=True)
        # Empath, plutchik
        self.empth = Empath()
        # NRC, plutchik
        self.nrc = pd.read_csv(join(self.lexicons_path, 'nrc.txt'), sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()
        self.nrc.rename(columns={'negative': 'negative_emotion', 'positive': 'positive_emotion'}, inplace=True)
        del self.nrc['positive_emotion']
        del self.nrc['negative_emotion']

        # print(self.nrc [self.nrc.word == str('cry')])

        # SentiSense
        self.senti_Sense = pd.read_csv(join(self.lexicons_path, 'senti_sense.csv'), sep=',', names=['word', 'emotion'], header=None)
        self.senti_Sense['value'] = 1
        self.senti_Sense = pd.pivot_table(self.senti_Sense, index='word', columns='emotion', values='value', fill_value=0).reset_index()
        # LIWC, sad, anger, neg & pos emotion
        self.liwc = readDict(join(self.lexicons_path, 'liwc.dic'))
        self.liwc = pd.DataFrame(self.liwc, columns=['word', 'category'])
        self.liwc['word'] = self.liwc['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.liwc['value'] = 1
        self.liwc = pd.pivot_table(self.liwc, index='word', columns=['category'],
                                   values='value', fill_value=0).reset_index().reindex(['word', 'posemo', 'negemo', 'sad', 'anger'], axis=1)
        self.liwc.rename(columns={'negemo': 'negative_emotion', 'posemo': 'positive_emotion', 'sad': 'sadness'}, inplace=True)

    def normalize(self, x, by_value):
        return np.array(x).reshape(1, -1)#np.divide(x, by_value).reshape(1,-1)

    def lex_emo_senti_net(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, self.emoSentiNet[self.emoSentiNet.Concepts == str(word)].values.tolist()[0][1:]))
            except:
                pass
        return self.normalize(result, len(splitted_sentence))

    def lex_empath(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, list(self.empth.analyze(word,
                           categories=['Trust', 'surprise', 'positive_emotion', 'negative_emotion', 'anticipation',
                                       'sadness', 'joy', 'fear', 'disgust', 'angry']).values())))
            except:
                pass
        return self.normalize(result, len(splitted_sentence))

    def lex_NRC(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0, 0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, self.nrc[self.nrc.word == str(word)].values.tolist()[0][1:]))
            except:
                pass
        return self.normalize(result, len(splitted_sentence))

    def lex_senti_sense(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, self.senti_Sense[self.senti_Sense.word == str(word)].values.tolist()[0][1:]))
            except:
                pass
        return self.normalize(result, len(splitted_sentence))

    def lex_liwc(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, self.liwc[self.liwc.word == str(word)].values.tolist()[0][1:]))
            except:
                pass
        return self.normalize(result, len(splitted_sentence))


    def one_vector_emo(self, sentence):
        global_vec = []
        global_vec.extend(self.lex_emo_senti_net(sentence).tolist()[0])
        global_vec.extend(self.lex_empath(sentence).tolist()[0])
        global_vec.extend(self.lex_NRC(sentence).tolist()[0])
        global_vec.extend(self.lex_senti_sense(sentence).tolist()[0])
        global_vec.extend(self.lex_liwc(sentence).tolist()[0])
        return global_vec

    def increase_emo(self, key, value):
        if key in self.emotions and value > 0:
            self.emotions[key] = int(self.emotions[key]) + int(value)

    def aggregated_vector_emo(self, sentence, binary=False):
        splitted_sentence = str(sentence).lower().split()
        for word in splitted_sentence:
            # emoSentiNet
            result_emo = self.emoSentiNet[self.emoSentiNet.Concepts == str(word)]
            if len(result_emo) > 0:
                for column in result_emo:
                    self.increase_emo(column, result_emo[column].tolist()[0])

            # empath
            result_emo = self.empth.analyze(word,
                           categories=['trust', 'surprise', 'positive_emotion', 'negative_emotion', 'anticipation',
                                       'sadness', 'joy', 'fear', 'disgust', 'angry'])
            for key in result_emo:
                self.increase_emo(key, result_emo[key])

            # NRC
            result_emo = self.nrc[self.nrc.word == str(word)]
            if len(result_emo) > 0:
                for column in result_emo:
                    self.increase_emo(column, result_emo[column].tolist()[0])

            # sentSense
            result_emo = self.senti_Sense[self.senti_Sense.word == str(word)]
            if len(result_emo) > 0:
                for column in result_emo:
                    self.increase_emo(column, result_emo[column].tolist()[0])

            # LIWC
            result_emo = self.liwc[self.liwc.word == str(word)]
            if len(result_emo) > 0:
                for column in result_emo:
                    self.increase_emo(column, result_emo[column].tolist()[0])

        if binary:
            for key, value in self.emotions.items():
                if value > 1:
                    self.emotions[key] = 1
        return list(self.emotions.values())

