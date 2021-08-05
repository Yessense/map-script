import dataclasses
from dataclasses import field
from typing import List, Any, Tuple, Dict, Set
from enum import Enum
import re
import itertools
from nltk.stem.wordnet import WordNetLemmatizer

from src.script_extraction.text_preprocessing.words_object import Roles, POS, Action, Position, Obj, RESTRICTED_POS, \
    WordsObject
from src.text_info_cinema import create_text_info_cinema

lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet as wn  # type:
from nltk.wsd import lesk

# from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info

def process_action(action_info : Dict, pos_list: List,
                   words: List[str], sentence_number: int) -> Action:
    # regex patterns to deal with BIO notation
    start_arg = re.compile("(B)-(.*)")
    inside_arg = re.compile("(I)-(.*)")

    # action to add roles
    action: Action = Action(text=action_info['verb'])

    # sliding over sentence and finding arguments
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

            # add founded info to verb or role
            if Roles(argument_type) == Roles.V:
                action.position = position
            else:
                obj = Obj(text=text.lower(),
                          position=position,
                          arg_type=Roles(argument_type))
                obj.set_part_of_speech(pos_list=pos_list)
                if is_accepted(obj.pos):
                    action.add_obj(obj=obj)
        else:
            current_symbol += len(words[i]) + 1
            i += 1
    return action


def add_hypernims(actions: List[Action], text_info: Dict[str, Any]):
    # allowed_pos_types = {'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
    # for action in actions:
    #     for obj in action.objects:
    #         if obj.pos.value in allowed_pos_types:
    #             sent = text_info['sentences_info'][action.position.sentence_number]['semantic_roles']['words']
    #             ss = lesk(sent, obj.text)
    #             for hypernym in ss.hypernyms():
    #                 obj.hypernyms.update(hypernym.lemma_names())
    return actions


def extract_actions(text_info):
    actions: List[Action] = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        for action_info in sentence_info['semantic_roles']['verbs']:
            action = process_action(action_info=action_info,
                                    pos_list=sentence_info['dependency']['pos'],
                                    words=sentence_info['semantic_roles']['words'],
                                    sentence_number=i)
            actions.append(action)
    actions = resolve_phrases(actions, text_info)
    actions = add_hypernims(actions, text_info)
    # actions = assemble_actions(text_info, actions)
    return actions


def assemble_actions(text_info: Dict[str, Any],
                     actions: List[Action]) -> List[Action]:
    actions_dict = {action.index: action for action in actions}

    root_list: List[Action] = [Action() for _ in range(len(text_info['sentences_info']))]

    for i, sentence_info in enumerate(text_info['sentences_info']):
        find_actions(sentence_number=i,
                     node=sentence_info['dependency']['hierplane_tree']['root'],
                     actions_dict=actions_dict,
                     parent=root_list[i])

    return list(itertools.chain(*[action.actions for action in root_list]))


def find_actions(sentence_number, node: Dict[str, Any], actions_dict, parent: Action) -> None:
    """
    Find and add child actions to each action
    """
    index = (sentence_number, node['spans'][0]['start'], node['spans'][0]['end'])
    if index in actions_dict:
        if parent is not None:
            parent.add_action(actions_dict[index])
        parent = actions_dict[index]
    if 'children' in node:
        for child in node['children']:
            find_actions(sentence_number, child, actions_dict, parent)


def is_accepted(pos: POS):
    """Check pos  candidate"""
    return pos not in RESTRICTED_POS


def select_from_candidates(candidates_for_obj: List[Tuple[int, WordsObject]], min_level):
    obj = sorted([image for level, image in candidates_for_obj if level == min_level], key=lambda x: x.pos)[0]
    return obj


def select_images(candidates_for_obj, new_obj, min_level):
    images = [image for level, image in candidates_for_obj if
              image.position != new_obj.position]  # and level == min_level]
    return images


def resolve_phrases(actions: List[Action], text_info: Dict[str, Any]):
    # convert trees to list
    trees_list = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        trees_list.append(get_roots(sentence_info['dependency']['hierplane_tree']['root'],
                                    sentence_number=i))

    for action_index, action in enumerate(actions):
        for i, obj in enumerate(action.objects):
            if obj.pos == POS.PHRASE:
                candidates_for_obj: List[Tuple[int, WordsObject]] = [(level, image)
                                                               for level, image in
                                                               trees_list[obj.position.sentence_number]
                                                               if image.position.inside(obj.position) and is_accepted(
                        image.pos)]
                if not len(candidates_for_obj):
                    obj = None
                    continue

                min_level = min([level for level, image in candidates_for_obj])
                # select candidate
                new_obj: WordsObject = select_from_candidates(candidates_for_obj, min_level)
                obj.pos = new_obj.pos
                obj.text = new_obj.text
                obj.images = select_images(candidates_for_obj, new_obj, min_level)
        action.objects = [obj for obj in action.objects if obj is not None]
    return actions


def get_roots(node, sentence_number, words_list=None, level=0) -> List[Tuple]:
    if words_list is None:
        words_list = []
    words_list.append((level,
                       WordsObject(text=node['word'].lower(),
                             pos=POS(node['attributes'][0]),
                             position=Position(start_symbol=node['spans'][0]['start'],
                                               end_symbol=node['spans'][0]['end'],
                                               sentence_number=sentence_number))))
    if 'children' in node:
        for child in node['children']:
            get_roots(child, sentence_number, words_list, level + 1)
    return words_list


def example_usage():
    # text_info
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt'
    # text_info = extract_texts_info([filename])[0]
    text_info = create_text_info_cinema()

    actions = extract_actions(text_info)
    print("DONE")


if __name__ == '__main__':
    example_usage()
