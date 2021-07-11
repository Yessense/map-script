from src.script_extraction.text_preprocessing.role import Role


class Verb(Role):
    def __init__(self, argument_type, sentence_number, word_spans, string, cluster=None):
        super().__init__(argument_type, sentence_number, word_spans, string, cluster)
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
