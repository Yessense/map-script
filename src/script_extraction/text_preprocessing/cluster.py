"""
Coreference cluster and his elements
"""


class Cluster:
    """
    One cluster of words with refer to one thing
    """
    __index = 0

    def __init__(self):
        self.index = Cluster.__index
        Cluster.__index += 1
        self.elements = []
        self.argument_type = 'cluster'


    def add_element(self, element):
        self.elements.append(element)

    def __repr__(self):

        return f'{self.index} {set([element.string for element in self.elements]).__repr__()}'


class Element:
    """
    Element of cluster - one word with sentence, spans and string
    """

    def __init__(self, sentence_number, word_spans, string):
        self.sentence_number = sentence_number
        self.word_spans = word_spans
        self.string = string

    def index(self):
        return self.sentence_number, self.word_spans

    def __repr__(self):
        return {'sentence_number': self.sentence_number,
                'word_spans': self.word_spans,
                'string': self.string}.__repr__()
