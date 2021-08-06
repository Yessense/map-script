from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Any, List, Dict
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class Roles(Enum):
    """
    Semantic roles enum
    """
    ARG0 = 'ARG0'
    ARG1 = 'ARG1'
    ARG2 = 'ARG2'
    ARG3 = 'ARG3'
    ARG4 = 'ARG4'
    ARGM_TMP = 'ARGM-TMP'
    ARGM_DIR = 'ARGM-DIR'
    ARGM_DIS = 'ARGM-DIS'
    ARGM_EXT = 'ARGM-EXT'
    ARGM_LOC = 'ARGM-LOC'
    ARGM_MNR = 'ARGM-MNR'
    ARGM_MOD = 'ARGM-MOD'
    ARGM_NEG = 'ARGM-NEG'
    ARGM_PRD = 'ARGM-PRD'
    ARGM_PRP = 'ARGM-PRP'
    ARGM_ADV = 'ARGM-ADV'
    R_ARGM_TMP = 'R-ARGM-TMP'
    ARGM_GOL = 'ARGM-GOL'
    C_ARG1 = 'C-ARG1'
    R_ARG0 = 'R-ARG0'
    ARGM_CAU = 'ARGM-CAU'
    ARGM_REC = 'ARGM-REC'
    R_ARG1 = 'R-ARG1'
    V = 'V'
    NONE = "NONE"


class POS(Enum):
    """
    Part of speech enum
    """
    ADJ = 'ADJ'  # adjective
    ADP = 'ADP'  # adposition
    ADV = 'ADV'  # adverb
    AUX = 'AUX'  # auxiliary
    CCONJ = 'CCONJ'  # coordinating conjunction
    DET = 'DET'  # determiner
    INTJ = 'INTJ'  # interjection
    NOUN = 'NOUN'  # noun
    NUM = 'NUM'  # numeral
    PART = 'PART'  # particle
    PRON = 'PRON'  # pronoun
    PROPN = 'PROPN'  # proper noun
    PUNCT = 'PUNCT'  # punctuation
    SCONJ = 'SCONJ'  # subordinating conjunction
    SYM = 'SYM'  # symbol
    VERB = 'VERB'  # verb
    X = 'X'  # other
    PHRASE = 'PHRASE'
    NONE = 'NONE'

    # Требуется при сравнении и выборе слова из дерева
    # для разбора фразы
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            if self.value is self.NOUN and other.value is not POS.NOUN:
                return True
            return self.value < other.value
        return NotImplemented


# Restricted parts of speech for role filling
RESTRICTED_POS = {POS.PUNCT, POS.CCONJ, POS.DET, POS.ADP, POS.VERB, POS.PART}

# Part of speech mapping for lemmatizing
POS_FOR_LEM = {POS.ADJ: 'a', POS.ADV: 'r', POS.NOUN: 'n', POS.VERB: 'v'}


@dataclass
class Position:
    """
    Position of word or phrase in text
    """
    sentence_number: int = 0
    start_word: int = 0
    end_word: int = 0
    start_symbol: int = 0
    end_symbol: int = 0

    @property
    def index(self) -> Tuple[int, int, int]:
        return self.sentence_number, self.start_symbol, self.end_symbol

    @property
    def words(self) -> int:
        """
        Number of words on this position
        Returns
        -------
        out : int
            number of words in sentence

        """
        return self.end_word - self.start_word

    def set_end_symbol(self, text: str) -> None:
        self.end_symbol = self.start_symbol + len(text) + 1

    def inside(self, position: Any) -> bool:
        """
        Check if position is inside other position

        Parameters
        ----------
        position: Position

        Returns
        -------
        out : bool
            True if inside, False otherwise

        """

        return self.start_symbol >= position.start_symbol and self.end_symbol <= position.end_symbol


@dataclass
class WordsObject:
    """
    Base class for any words objects
    """
    text: str = ""
    position: Position = Position()
    pos: POS = POS.NONE

    @property
    def index(self) -> Tuple[int, int, int]:
        return self.position.index

    @property
    def lemma(self) -> str:
        # self.pos != POS.PHRASE
        if self.pos in POS_FOR_LEM:
            # more correct lematize, based on part of speech
            return lemmatizer.lemmatize(self.text, POS_FOR_LEM[self.pos])
        else:
            return lemmatizer.lemmatize(self.text)

    def set_part_of_speech(self, sentences_info: List[Dict[str, Any]]):
        if self.position.words == 1:
            self.pos = POS(sentences_info[self.position.sentence_number]['dependency']['pos'][self.position.start_word])
        else:
            self.pos = POS.PHRASE


@dataclass
class Obj(WordsObject):
    """
    Contains information about object
    role type, images
    """

    arg_type: Roles = Roles.NONE
    images: List[WordsObject] = field(default_factory=list)


@dataclass
class Action(WordsObject):
    """
    Contains information for action sign
    Roles, objects, children Actions
    """
    objects: List[Obj] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)

    def __post_init__(self):
        self.pos = POS.VERB

    def add_obj(self, obj: Obj) -> None:
        self.objects.append(obj)


@dataclass
class Cluster:
    """
    Contains all named group entities
    """
    named_group: List[WordsObject] = field(default_factory=list)

    def add_words_object(self, words_object: WordsObject) -> None:
        """

        Parameters
        ----------
        words_object

        Returns
        -------
        out: None

        """
        self.named_group.append(words_object)
