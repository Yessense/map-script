class Cluster:
    __index = 0
    def __init__(self):
        self.index = Cluster.__index
        Cluster.__index += 1
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def __repr__(self):
        return self.index.__repr__() + self.elements.__repr__()

class Element:
    def __init__(self, sentence_number, word_spans, string):
        self.sentence_number = sentence_number
        self.word_spans = word_spans
        self.string = string

    def __repr__(self):
        return {'sentence_number': self.sentence_number,
                'word_spans': self.word_spans,
                'string': self.string}.__repr__()

