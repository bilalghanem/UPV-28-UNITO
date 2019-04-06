import pandas as pd
from os.path import join
import ufal.udpipe, conllu, os
import features.syntactical.en_core_web_sm as en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer


class Model_udpipe:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()
        return output

class syntactic_BOW_class:
    def __init__(self, udpipe=False):
        self.udpipe = udpipe
        all_relations = ['vocative', 'predet', 'aux', 'ccomp', 'neg', 'remnant', 'advcl', 'nsubj', 'poss', 'mwe', 'reparandum', 'cop', 'case', 'conj', 'compound', 'det', 'goeswith', 'list', 'prep', 'amod', 'name', 'oprd', 'xcomp', 'relcl', 'nummod', 'appos', 'acl', 'advmod', 'nmod', 'discourse', 'root', 'prt', 'agent', 'meta', 'dobj', 'auxpass', 'expl', 'npadvmod', 'punct', 'ROOT', 'attr', 'csubjpass', 'quantmod', 'dative', 'pobj', 'pcomp', 'intj', 'dep', 'iobj', 'cc', 'acomp', 'dislocated', 'parataxis', 'mark', 'preconj', 'nsubjpass', 'csubj', 'foreign']
        all_udpipe_relations = ['nsubj','obj','iobj','csubj','ccomp','xcomp','obl','vocative','expl','dislocated','advcl','advmod','discourse','aux','cop','mark','nmod','appos','nummod','acl','amod','det','clf','case','conj','cc','fixed','flat','compound','list','parataxis','orphan','goeswith','reparandum','punct','root','dep']
        self.count_vec = CountVectorizer()
        if udpipe:
            self.count_vec.fit(all_udpipe_relations)
        else:
            self.count_vec.fit(all_relations)
        self.cue_words_list = []
        current = join(os.path.dirname(os.path.realpath(__file__)), 'cue_words')
        for filename in os.listdir(current):
            text_file = open(join(current, filename), "r")
            self.cue_words_list.extend(text_file.read().split('\n'))
            text_file.close()

        self.nlp = en_core_web_sm.load()
        self.udpipe_model = Model_udpipe('features/syntactical/english-ud-1.2-160523.udpipe')

    def score(self, sentence):
        sentence_relations = []
        if self.udpipe == False:
            doc = self.nlp(sentence)
            for token in doc:
                sentence_relations.append(str(token.dep_))
            BOR = self.count_vec.transform([' '.join(sentence_relations)]).toarray().tolist()
            # BOR = BOR.tolist()
        else:
            sentences = self.udpipe_model.tokenize(sentence)
            for s in sentences:
                self.udpipe_model.tag(s)
                self.udpipe_model.parse(s)
            conllu_txt = self.udpipe_model.write(sentences, "conllu") #conllu|horizontal|vertical
            conllu_obj = conllu.parse(conllu_txt)
            for item in conllu_obj[0]:
                sentence_relations.append(str(item['deprel']))
            BOR = self.count_vec.transform([' '.join(sentence_relations)]).toarray().tolist()
        return BOR[0]

    def score_words(self, sentence, from_list=False):
        sentences = self.udpipe_model.tokenize(sentence)
        for s in sentences:
            self.udpipe_model.tag(s)
            self.udpipe_model.parse(s)
        conllu_txt = self.udpipe_model.write(sentences, "conllu") #conllu|horizontal|vertical
        conllu_obj = conllu.parse(conllu_txt)

        if not from_list:
            words = ['agree', 'disagree']
        else:
            words = self.cue_words_list
        words_id = []
        relations = []
        for word in conllu_obj[0]:
            if word['form'] in words:
                words_id.append(word['id'])
        if len(words_id):
            for word in conllu_obj[0]:
                if word['head'] in words_id or word['id'] in words_id:
                    relations.append(word['deprel'])

        BOR = self.count_vec.transform([' '.join(relations)]).toarray().tolist()
        return BOR[0]

    def score_verbs(self, sentence):
        sentences = self.udpipe_model.tokenize(sentence)
        for s in sentences:
            self.udpipe_model.tag(s)
            self.udpipe_model.parse(s)
        conllu_txt = self.udpipe_model.write(sentences, "conllu") #conllu|horizontal|vertical
        conllu_obj = conllu.parse(conllu_txt)

        words_id = []
        relations = []
        for word in conllu_obj[0]:
            if str(word['upostag']).lower() == 'verb':
                words_id.append(word['id'])
        if len(words_id):
            for word in conllu_obj[0]:
                if word['head'] in words_id or word['id'] in words_id:
                    relations.append(word['deprel'])

        BOR = self.count_vec.transform([' '.join(relations)]).toarray().tolist()
        return BOR[0]

if __name__ == '__main__':
    h = syntactic_BOW_class()
    print(h.score("the apple wasn't eaten by simona"))
