import dataclasses
from dataclasses import field
from typing import List, Any, Tuple
from enum import Enum
import re

from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info


class Roles(Enum):
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
    V = 'V'


class POS(Enum):
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


@dataclasses.dataclass
class Position:
    sentence_number: int = 0
    start_word: int = 0
    end_word: int = 0
    start_symbol: int = 0
    _end_symbol: int = 0

    def words(self):
        return self.end_word - self.start_word

    @property
    def end_symbol(self) -> int:
        return self._end_symbol

    @end_symbol.setter
    def end_symbol(self, text: str) -> None:
        self._end_symbol = self.start_symbol + len(text) + 1


@dataclasses.dataclass
class Image:
    position: Position
    text: str


@dataclasses.dataclass
class Obj:
    """
    Contains information about object
    role type, images
    """
    text: str
    position: Position
    arg_type: Roles
    pos: POS = POS.PHRASE
    images: List[Image] = field(default_factory=list)

    def set_part_of_speech(self, pos_list):
        if self.position == 1:
            self.pos = POS(pos_list[self.position.start_word])


@dataclasses.dataclass
class Action:
    """
    Contains information for action sign
    Roles, objects, children Actions
    """
    text: str
    position: Position = Position()
    roles: List[Obj] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    arg_type: Roles = Roles.V

    def add_obj(self, obj: Obj) -> None:
        self.roles.append(obj)


def process_action(action_info, pos_list, words, sentence_number):
    # regex patterns to deal with BIO notation
    start_arg = re.compile("(B)-(.*)")
    inside_arg = re.compile("(I)-(.*)")

    # action to add roles
    action: Action = Action(text=action_info['verb'])

    i: int = 0
    sentence_len = len(action_info['tags'])
    current_symbol = 0
    while i < sentence_len:
        # if tag is begin of arg
        if (match := re.fullmatch(start_arg, action_info['tags'][i])) is not None:

            # start processing one argument
            text = words[i]
            position = Position(sentence_number=sentence_number,
                                start_word=i,
                                end_word=i + 1,
                                start_symbol=current_symbol)

            # find argument type
            argument_type_spans = match.regs[2]
            argument_type: str = action_info['tags'][i][argument_type_spans[0]:argument_type_spans[1]]

            current_symbol += len(words[i]) + 1
            # iterate through tags in search of current argument in-tags
            i += 1
            while i < sentence_len and re.fullmatch(inside_arg, action_info['tags'][i]) is not None:
                text += " " + words[i]

                current_symbol += len(words[i]) + 1
                i += 1
                position.end_word = i

            position.end_symbol(text)

            # add info to verb or role
            if Roles(argument_type) == Roles.V:
                action.position = position
            else:
                role = Obj(text=text,
                           position=position,
                           arg_type=Roles(argument_type))
                role.set_part_of_speech()
                action.add_obj(role)
        else:
            current_symbol += len(words[i]) + 1
            i += 1
    return action


def extract_semantic_roles(sentence_info, sentence_number):
    semantic_roles = []
    for action_info in sentence_info['semantic_roles']['verbs']:
        action = process_action(action_info=action_info,
                                pos_list=sentence_info['dependency']['pos'],
                                words=sentence_info['semantic_roles']['words'],
                                sentence_number=sentence_number)
        semantic_roles.append(action)
    return semantic_roles


def example_usage():
    # text_info
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    actions = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        actions += extract_semantic_roles(sentence_info, i)
    print("DONE")


if __name__ == '__main__':
    example_usage()
