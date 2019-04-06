import pandas as pd
from os.path import join


class hurt_lex_class:
    def __init__(self, path=''):
        self.hurt_lex = pd.read_csv(join(path, 'hurtlex_EN_conservative.tsv'), sep='\t', names=["category", "_", "words"])
        print('')

    def score(self, sentence):
        words = sentence.split()
        or_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'or']['words'].tolist())))
        an_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'an']['words'].tolist())))
        asm_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'asm']['words'].tolist())))
        asf_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'asf']['words'].tolist())))
        qas_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'qas']['words'].tolist())))
        cds_ = len(set(words).intersection(set(self.hurt_lex[self.hurt_lex['category'] == 'cds']['words'].tolist())))
        return [or_, an_, asm_, asf_, qas_, cds_]

if __name__ == '__main__':
    sentence = "I'm wondering, why there are not student here? bad fuck suck"
    h = hut_lex_class()
    print(h.score(sentence))
