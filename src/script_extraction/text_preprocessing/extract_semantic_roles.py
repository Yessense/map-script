from src.script_extraction.text_preprocessing.role import Role
from tests.get_info import get_text_info
import re

start_arg = re.compile("(B)-(.*)")
inside_arg = re.compile("(I)-(.*)")


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

    for verb_info in sentence_info['semantic role']['verbs']:
        verb = process_verb(verb_info, sentence_info['semantic role']['words'], sentence_number)
        semantic_roles.append(verb)

    return semantic_roles


def example_usage():
    # text_info
    text_info = get_text_info()

    roles = extract_semantic_roles(text_info['sentences_info'][0], 5)
    print("DONE")


if __name__ == '__main__':
    example_usage()
