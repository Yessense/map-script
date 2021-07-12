class Role:
    def __init__(self, argument_type, sentence_number, word_spans, text, cluster=None):
        self.type = argument_type
        self.sentence_number = sentence_number
        self.word_number = word_spans
        self.text = text
        self.cluster = cluster

    def set_cluster(self, value):
        self.cluster = value

    def __repr__(self):
        return f"{self.type}:{self.text}"




