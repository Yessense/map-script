import dataclasses
from dataclasses import field
from typing import List, Any, Tuple
from enum import Enum
import re

from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info

start_arg = re.compile("(B)-(.*)")
inside_arg = re.compile("(I)-(.*)")


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

@dataclasses.dataclass
class Position:
    sentence_number: int
    start_word: int
    end_word: int


@dataclasses.dataclass
class Image:
    position: Any
    text: str


@dataclasses.dataclass
class Role:
    text: str
    position: Position
    pos: str
    images: List[Image]
    arg_type: Roles


@dataclasses.dataclass
class Action:
    text: str
    position: Position
    roles: List[Role] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    arg_type: Roles = Roles.V


def process_action(action_info, words, sentence_number):
    # action to add roles
    action = Action(text=action_info['verb'],
                    sentence_number=sentence_number)


def process_verb(verb_info, words, sentence_number):
    # verb to add all roles
    verb = Role(text=verb_info['verb'],
                sentence_number=sentence_number,
                argument_type='V')

    # iterate through all words
    i: int = 0
    sentence_len = len(verb_info['tags'])
    while i < sentence_len:

        # if tag is begin of arg
        if (match := re.fullmatch(start_arg, verb_info['tags'][i])) is not None:

            # start processing one argument
            text = words[i]
            start_pos = i
            end_pos = i

            # find argument type
            argument_type_spans = match.regs[2]
            argument_type = verb_info['tags'][i][argument_type_spans[0]:argument_type_spans[1]]

            # iterate through tags in search of current argument in-tags
            i += 1
            while i < sentence_len and re.fullmatch(inside_arg, verb_info['tags'][i]) is not None:
                text += " " + words[i]
                end_pos = i
                i += 1

            # if it is verb itself, fiil verb, otherwise create role and add to verb
            if argument_type == 'V':
                verb.words_spans = (start_pos, end_pos + 1)
            else:
                role = Role(argument_type=argument_type,
                            sentence_number=sentence_number,
                            words_spans=(start_pos, end_pos + 1),
                            text=text)
                verb.add_role(role)
        else:
            i += 1
    return verb


def extract_semantic_roles(sentence_info, sentence_number):
    semantic_roles = []

    for verb_info in sentence_info['semantic_roles']['verbs']:
        verb = process_verb(verb_info, sentence_info['semantic_roles']['words'], sentence_number)
        semantic_roles.append(verb)

    return semantic_roles


def example_usage():
    # text_info
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    roles = extract_semantic_roles(text_info['sentences_info'][0], 5)
    print("DONE")


if __name__ == '__main__':
    example_usage()
