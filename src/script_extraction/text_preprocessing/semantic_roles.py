import re

from tests.get_info import get_text_info


def process_sentence(sentence_info):
    """
    Add semantic roles to hierplane tree

    Parameters
    ----------
    sentence_info: dict of dict

    Returns
    -------
    dict of dict
    """


def add_roles(node, roles):
    """
    Recurcive search verb to append roles
    Parameters
    ----------
    node: dict
    roles: dict

    Returns
    -------
    None

    """
    if (node['spans'][0]['start'] == roles['V']['span'][0]
            and node['spans'][0]['end'] == roles['V']['span'][1]):
        node['roles'] = roles
        return
    else:
        if 'children' in node:
            for child in node['children']:
                add_roles(child, roles)
            return
        else:
            return


def add_semantic_roles(text_info):
    """
    Add semantic roles dict to each dependency tree

    Parameters
    ----------
    text_info : dict of dict

    Returns
    -------
    out: dict of dict
    """
    arg_regex = re.compile("\[(.+?): (.+?)\]")

    for sentence_info in text_info['sentences_info']:
        for verb in sentence_info['semantic role']['verbs']:
            sentence = verb['description']

            roles = {}
            start = 0
            real_sentence_pos_shift = 0

            while arg := arg_regex.search(sentence, start):
                # spans for all groups in regex
                match_object_span, arg_type_span, word_span = arg.regs

                # get words
                role = sentence[arg_type_span[0]:arg_type_span[1]]
                word = sentence[word_span[0]:word_span[1]]

                real_sentence_pos_shift += match_object_span[1] - match_object_span[0] - (word_span[1] - word_span[0])

                # add to dict
                roles[role] = {'word': word}
                # shift word pos
                roles[role]['span'] = word_span[0] - real_sentence_pos_shift + 1, word_span[1] - real_sentence_pos_shift + 2

                start = arg.regs[0][1]
            add_roles(sentence_info['dependency']['hierplane_tree']['root'], roles)
    return text_info

def example_usage():
    text_info = get_text_info()

    new_text_info = add_semantic_roles(text_info)

    print("DONE")




if __name__ == '__main__':
    example_usage()