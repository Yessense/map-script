from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Any, List, Dict, Union, Optional, Set
from nltk.stem.wordnet import WordNetLemmatizer

from src.script_extraction.wn import get_meaning

lemmatizer = WordNetLemmatizer()


# ----------------------------------------
# Enums
# ----------------------------------------

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
    NAMED_GROUP = "NAMED-GROUP"


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
ALLOWED_POS = {POS.VERB, POS.NOUN, POS.ADV, POS.ADJ, POS.PRON, POS.PHRASE}
# Allowed pronouns
ALLOWED_PRON = {"i", "we"}
# Part of speech mapping for lemmatizing
POS_FOR_LEM = {POS.ADJ: 'a', POS.ADV: 'r', POS.NOUN: 'n', POS.VERB: 'v'}


# ----------------------------------------
# Classes
# ----------------------------------------

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

    def set_symbols_bounds(self, sentence, text):
        start_symbol = 0
        for word_number in range(self.start_word):
            start_symbol += len(sentence[word_number]) + 1

        self.start_symbol = start_symbol
        self.set_end_symbol(text)

    def set_words_bounds(self, sentence: List[str]):
        # finding start word
        current_symbol = 0
        for word_number, word in enumerate(sentence):
            if self.start_symbol == current_symbol:
                self.start_word = word_number
                break
            else:
                current_symbol += len(word) + 1

        # finding end word
        current_symbol = self.start_symbol
        for word_number, word in enumerate(sentence[self.start_word:]):
            current_symbol += len(word) + 1
            if self.end_symbol == current_symbol:
                self.end_word = self.start_word + word_number + 1
                break





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
    _synsets_len: int = 1
    _synset_number: int = 0
    _lemma: Optional[str] = None
    cluster: Optional[Any] = None

    def __post_init__(self):
        self.text = self.text.lower()

    @property
    def index(self) -> Tuple[int, int, int]:
        return self.position.index

    @property
    def lemma(self) -> str:
        if self._lemma is not None:
            return self._lemma
        # self.pos != POS.PHRASE
        if self.pos in POS_FOR_LEM:
            # more correct lematize, based on part of speech
            self._lemma = lemmatizer.lemmatize(self.text, POS_FOR_LEM[self.pos])
        else:
            self._lemma = lemmatizer.lemmatize(self.text)
        return self._lemma

    def set_part_of_speech(self, sentences_info: List[Dict[str, Any]]):
        if self.position.words == 1:
            self.pos = POS(sentences_info[self.position.sentence_number]['dependency']['pos'][self.position.start_word])
        else:
            self.pos = POS.PHRASE

    @property
    def is_accepted(self) -> bool:
        """Check pos  candidate"""
        return self.pos in ALLOWED_POS

    def set_meaning(self, text_info: Dict[str, Any]) -> None:
        sentence = text_info['sentences_info'][self.position.sentence_number]['semantic_roles']['words']

        self._synsets_len, self._synset_number = get_meaning(sentence=sentence,
                                                             lemma=self.lemma,
                                                             pos=POS_FOR_LEM.get(self.pos, None))

    @property
    def synsets_len(self):
        return self._synsets_len

    @property
    def synset_number(self):
        return self._synset_number


@dataclass
class Obj(WordsObject):
    """
    Contains information about object
    role type, images
    """
    arg_type: Roles = Roles.NONE
    images: List[Any] = field(default_factory=list)


@dataclass
class Action(WordsObject):
    """
    Contains information for action sign
    Roles, objects, children Actions
    """
    objects: List[Obj] = field(default_factory=list)

    def __post_init__(self):
        self.pos = POS.VERB

    def add_obj(self, obj: Obj) -> None:
        self.objects.append(obj)


@dataclass
class Cluster:
    """
    Contains all named group entities
    """
    # named group - allenlp coreferences out
    objects: List[Obj] = field(default_factory=list)
    # according to cluster objects, real objects
    real_objects: List[Union[WordsObject, Obj, Action]] = field(default_factory=list)
    images: Dict[Tuple[int, int, int], WordsObject] = field(default_factory=dict)

    def add_cluster_obj(self, obj: Obj) -> None:
        self.objects.append(obj)
        for image in obj.images:
            self.add_image(obj)

    def add_image(self, obj: Union[WordsObject, Obj, Action]) -> None:
        if obj.index not in self.images:
            self.images[obj.index] = obj

    def add_real_obj(self, obj: Union[WordsObject, Obj, Action]) -> None:
        self.real_objects.append(obj)
        if isinstance(obj, Obj):
            for image in obj.images:
                self.add_image(image)
