import warnings
warnings.filterwarnings("ignore")
from features.syntactical.syntactic_negation.syntactic_negation import syntactic_negation_class
from features.syntactical.syntactic_relations.syntactic_relations import syntactic_BOW_class


class Syntactic:

    def __init__(self, udpipe=False):
        self.s_netation = syntactic_negation_class(udpipe=udpipe)
        self.s_BOW = syntactic_BOW_class(udpipe=udpipe)

    def one_vector_synt(self, sentence):
        global_vec = []
        global_vec.extend(self.s_netation.score(sentence))
        global_vec.extend(self.s_BOW.score(sentence))
        global_vec.extend(self.s_BOW.score_words(sentence, from_list=True))
        global_vec.extend(self.s_BOW.score_verbs(sentence))

        return global_vec

if __name__ == '__main__':
    snt = Syntactic()
    s = ["I don't like the movie, it's bad I hate it", "I love you sweety kiss", 'I was so sad, he called me the bitch, he was killed']
    print(snt.one_vector_synt(s[0]))
