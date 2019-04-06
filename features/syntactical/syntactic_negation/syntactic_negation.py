import pandas as pd
from os.path import join
import ufal.udpipe, conllu
import features.syntactical.en_core_web_sm as en_core_web_sm


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

class syntactic_negation_class:
    def __init__(self, udpipe=False):
        self.udpipe = udpipe
        self.nlp = en_core_web_sm.load()
        self.udpipe_model = Model_udpipe('features/syntactical/english-ud-1.2-160523.udpipe')

    def score(self, sentence):
        negation_count = 0
        all_dep_count = 0

        if self.udpipe == False:
            doc = self.nlp(sentence)
            for token in doc:
                if str(token.dep_) == 'neg':
                    negation_count += 1
                else:
                    all_dep_count += 1
        else:
            sentences = self.udpipe_model.tokenize(sentence)
            for s in sentences:
                self.udpipe_model.tag(s)
                self.udpipe_model.parse(s)
            conllu_txt = self.udpipe_model.write(sentences, "conllu") #conllu|horizontal|vertical
            conllu_obj = conllu.parse(conllu_txt)
            for item in conllu_obj[0]:
                if str(item['deprel']) == 'neg':
                    negation_count += 1
                else:
                    all_dep_count += 1

        return [negation_count/all_dep_count]

if __name__ == '__main__':
    h = syntactic_negation_class()
    print(h.score("the apple wasn't eaten by simona"))
