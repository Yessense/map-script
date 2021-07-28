import dataclasses
from dataclasses import field
from typing import List, Any, Tuple, Dict
from enum import Enum
import re
import itertools

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
    ARGM_ADV = 'ARGM-ADV'
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
    end_symbol: int = 0

    def words(self):
        return self.end_word - self.start_word

    def set_end_symbol(self, text: str) -> None:
        self.end_symbol = self.start_symbol + len(text) + 1

    def get_dict_key(self):
        return self.sentence_number, self.start_symbol, self.end_symbol


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
    text: str = ""
    position: Position = Position()
    roles: List[Obj] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    arg_type: Roles = Roles.V

    def add_obj(self, obj: Obj) -> None:
        self.roles.append(obj)

    def add_action(self, action: Any) -> None:
        self.actions.append(action)


def process_action(action_info, pos_list, words, sentence_number) -> Action:
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

            position.set_end_symbol(text)

            # add info to verb or role
            if Roles(argument_type) == Roles.V:
                action.position = position
            else:
                obj = Obj(text=text,
                          position=position,
                          arg_type=Roles(argument_type))
                obj.set_part_of_speech(pos_list=pos_list)
                action.add_obj(obj=obj)
        else:
            current_symbol += len(words[i]) + 1
            i += 1
    return action


def extract_actions(text_info):
    actions: List[Action] = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        for action_info in sentence_info['semantic_roles']['verbs']:
            action = process_action(action_info=action_info,
                                    pos_list=sentence_info['dependency']['pos'],
                                    words=sentence_info['semantic_roles']['words'],
                                    sentence_number=i)
            actions.append(action)
    return actions


def assemble_actions(text_info: Dict[str, Any],
                     actions: List[Action]) -> List[Action]:
    actions_dict = {action.position.get_dict_key(): action for action in actions}

    root_list: List[Action] = [Action() for _ in range(len(text_info['sentences_info']))]

    for i, sentence_info in enumerate(text_info['sentences_info']):
        find_actions(sentence_number=i,
                     node=sentence_info['dependency']['hierplane_tree']['root'],
                     actions_dict=actions_dict,
                     parent=root_list[i])

    return list(itertools.chain(*[action.actions for action in root_list]))


def find_actions(sentence_number, node: Dict[str, Any], actions_dict, parent: Action) -> None:
    index = (sentence_number, node['spans'][0]['start'], node['spans'][0]['end'])
    if index in actions_dict:
        if parent is not None:
            parent.add_action(actions_dict[index])
        parent = actions_dict[index]
    if 'children' in node:
        for child in node['children']:
            find_actions(sentence_number, child, actions_dict, parent)


def example_usage():
    # text_info
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    actions = extract_actions(text_info)
    actions = assemble_actions(text_info, actions)
    print("DONE")


if __name__ == '__main__':
    example_usage()
