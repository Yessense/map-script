import dataclasses
from dataclasses import field
from typing import List, Any, Tuple, Dict, Set, Union
from enum import Enum
import re
import itertools
from nltk.stem.wordnet import WordNetLemmatizer

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.resolve_phrases import get_trees_list, resolve_phrases
from src.script_extraction.text_preprocessing.words_object import Roles, POS, Action, Position, Obj, \
    WordsObject, Cluster

from src.text_info_cinema import create_text_info_cinema
from src.text_info_restaurant import create_text_info_restaurant


def process_action(action_info: Dict, sentences_info: List[Dict[str, Any]],
                   words: List[str], sentence_number: int) -> Action:
    """
    Create action and his object

    Parameters
    ----------
    action_info
    sentences_info
    words
    sentence_number

    Returns
    -------

    """

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
                obj.set_part_of_speech(sentences_info=sentences_info)
                if obj.is_accepted:
                    action.add_obj(obj=obj)
        else:
            current_symbol += len(words[i]) + 1
            i += 1
    return action


def extract_actions(text_info: Dict) -> List[Action]:
    """
    Extract actions_signs from each sentence in text
    and return list of actions_signs
    Parameters
    ----------
    text_info

    Returns
    -------
    out: List[Action]

    """
    actions: List[Action] = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        for action_info in sentence_info['semantic_roles']['verbs']:
            action = process_action(action_info=action_info,
                                    sentences_info=text_info['sentences_info'],
                                    words=sentence_info['semantic_roles']['words'],
                                    sentence_number=i)
            actions.append(action)
    trees_list = get_trees_list(text_info)
    resolve_phrases(actions, trees_list=trees_list, text_info=text_info)
    # TODO: Add hypernyms
    return actions


def example_usage():
    # _text_info
    # filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt'
    # _text_info = extract_texts_info([filename])[0]
    text_info = create_text_info_restaurant()

    actions = extract_actions(text_info)
    print("DONE")


if __name__ == '__main__':
    example_usage()
