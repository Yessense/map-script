from src.script_extraction.text_preprocessing.role import Role


class Verb:
    def __init__(self, text: str, sentence_number=None, words_spans=None, cluster=None):
        self.text = text
        self.sentence_number = sentence_number
        self.words_spans = words_spans
        self.cluster = cluster
        self.roles = []

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

    def __repr__(self):
        return self.text.__repr__() + " " +  self.roles.__repr__()
