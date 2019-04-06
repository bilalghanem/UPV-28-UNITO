import warnings
warnings.filterwarnings("ignore")
from os.path import join
from features.lexical.bad_sexual_words.bad_sexual_words import bad_sexual_words_class
from features.lexical.cue_words.cue_words import cue_words_class
from features.lexical.hurtlex.hurt_lex import hurt_lex_class
from features.lexical.linguistic_words.linguistic_words import linguistic_words_class
from features.lexical.LIWC.LIWC import LIWC_class

warnings.filterwarnings("ignore")

class Lexical:

    def __init__(self, path=''):
        self.bad_sexual = bad_sexual_words_class(path=join(path, 'bad_sexual_words/'))
        self.cue_words = cue_words_class(path=join(path, 'cue_words/'))
        self.hurtlex = hurt_lex_class(path=join(path, 'hurtlex/'))
        self.linguistic_words = linguistic_words_class(path=join(path, 'linguistic_words/'))
        self.LIWC = LIWC_class(path=join(path, 'LIWC/'))


    def one_vector_lexical(self, sentence):
        global_vec = []
        sentence = str(sentence).lower()
        global_vec.extend(self.bad_sexual.score(sentence))
        global_vec.extend(self.cue_words.score(sentence))
        global_vec.extend(self.hurtlex.score(sentence))
        global_vec.extend(self.linguistic_words.score(sentence))
        global_vec.extend(self.LIWC.score(sentence))
        return global_vec

if __name__ == '__main__':
    snt = Lexical()
    s = ["I don't like the movie, it's bad I hate it", "I love you sweety kiss", 'I was so sad, he called me the bitch, he was killed']
    print(snt.one_vector_lexical(s[0]))
