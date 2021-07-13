from typing import List

class Role:
    def __init__(self, text: str,
                 sentence_number: int,
                 argument_type: str,
                 words_spans=None):
        self.text = text
        self.sentence_number = sentence_number
        self.argument_type = argument_type

        self.words_spans = words_spans

        self.roles : List[Role] = []

    def set_cluster(self, value):
        self.cluster = value

    def add_role(self, role):
        """

        Parameters
        ----------
        role : Role

        Returns
        -------
        out: None

        """
        self.roles.append(role)

    def index(self):
        return self.sentence_number, self.words_spans

    def label(self):
        return self.__repr__()

    def __repr__(self):
        return f"{(self.sentence_number, self.words_spans)} {self.text}"
