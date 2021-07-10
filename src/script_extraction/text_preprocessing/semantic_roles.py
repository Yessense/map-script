import re

from tests.get_info import get_text_info


def is_node_word_inside_spans(node, spans):
    """
    Find if node['spans'] inside spans

    Parameters
    ----------
    node: dict
    spans: tuple

    Returns
    -------
    bool
        True if inside, False otherwise
    """
    return (node['spans'][0]['start'] >= spans[0]
            and node['spans'][0]['end'] <= spans[1])


def add_roles(node, roles):
    """
    Recursive verb search to append roles
    Searching by matching spans

    Parameters
    ----------
    node: dict
    roles: dict

    Returns
    -------
    None

    """
    if is_node_word_inside_spans(node, roles['V']['spans']):
        node['roles'] = roles
        return
    else:
        if 'children' in node:
            for child in node['children']:
                add_roles(child, roles)
            return
        else:
            return


def get_roots(root, spans, words_list=None, level=0):
    """
    Find words inside spans and their tree height

    Parameters
    ----------
    root
    spans
    words_list
    level

    Returns
    -------
    out : list of (str, int, (int, int))
        [(word, tree height, (start, end))]
    """
    if words_list is None:
        words_list = []
    if is_node_word_inside_spans(root, spans):
        words_list.append((root['word'], level, (root['spans'][0]['start'], root['spans'][0]['end'])))
    if 'children' in root:
        for child in root['children']:
            get_roots(child, spans, words_list, level + 1)
    return words_list


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

    # pattern to find arguments in semantic role description
    # arguments looks like [ARG1: word]
    arg_regex = re.compile("\[(.+?): (.+?)\]")

    # looking in each sentence
    for sentence_info in text_info['sentences_info']:
        # root in current sentence for matching
        root = sentence_info['dependency']['hierplane_tree']['root']

        # finding roles for each founded verb
        for verb in sentence_info['semantic role']['verbs']:
            # string with all args description
            sentence = verb['description']

            # roles for current verb
            roles = {}

            # for shifting search
            search_position = 0

            # for shifting spans numbers, one shift for all sentence
            real_sentence_pos_shift = 0

            # looking for all args in description
            while arg := arg_regex.search(sentence, search_position):
                # spans for all groups in regex
                match_object_span, arg_type_span, word_span = arg.regs

                # update shift to substract from pos
                real_sentence_pos_shift += match_object_span[1] - match_object_span[0] - (word_span[1] - word_span[0])

                # get shifted word position (start, end)
                spans = word_span[0] - real_sentence_pos_shift + 1, word_span[1] - real_sentence_pos_shift + 2

                # [argument type : string]
                argument_type = sentence[arg_type_span[0]:arg_type_span[1]]

                # semantic role string may consist of more than one word
                string = sentence[word_span[0]:word_span[1]]

                # so we need to find closest to the root words in this string
                # parse hierplane tree to get roots from string
                root_words = get_roots(root, spans)

                # closest tree height value
                min_depth_value = min([word[1] for word in root_words])

                # writing closes words to words list
                roles[argument_type] = {
                    'words': [(word[0], word[2]) for word in root_words if word[1] == min_depth_value]}
                # writing their spans to next list
                roles[argument_type]['spans'] = spans

                if spans in sentence_info['coreferences']:
                    roles[argument_type]['cluster'] = sentence_info['coreferences'][spans]['cluster']

                # updating search position
                search_position = match_object_span[1]

            # add roles to verb
            add_roles(root, roles)
    return text_info


def add_sentences_bounds(text_info):
    number_of_previous_words = 0

    for sentence in text_info['sentences_info']:
        number_of_words_in_sentence = len(sentence['dependency']['words'])
        sentence['sentence_bounds'] = (
            number_of_previous_words, number_of_previous_words + number_of_words_in_sentence)
        number_of_previous_words += number_of_words_in_sentence


def add_words_bounds(text_info):
    for sentence in text_info['sentences_info']:
        words_bounds = []
        start_word_pos = 0
        for word in sentence['semantic role']['words']:
            words_bounds.append((start_word_pos, start_word_pos + len(word) + 1))
            start_word_pos += len(word) + 1
        sentence['words_bounds'] = words_bounds


def create_coreferences_clusters(text_info):
    add_sentences_bounds(text_info)
    add_words_bounds(text_info)
    clusters_info = []
    for cluster in text_info['coreferences']['clusters']:
        cluster_info = []
        for entry in cluster:
            for sentence_number, sentence in enumerate(text_info['sentences_info']):
                if entry[1] < sentence['sentence_bounds'][1] and entry[0] >= sentence['sentence_bounds'][0]:
                    start_word_number = entry[0] - sentence['sentence_bounds'][0]
                    end_word_number = entry[1] - sentence['sentence_bounds'][0]

                    start_symbol_number = sentence['words_bounds'][start_word_number][0]
                    end_symbol_number = sentence['words_bounds'][end_word_number][1]

                    entry_info = {'sentence_number': sentence_number,
                                  'words_spans': (start_word_number, end_word_number),
                                  'symbols_spans': (start_symbol_number, end_symbol_number),
                                  'string': sentence['dependency']['hierplane_tree']['text'][
                                            start_symbol_number:end_symbol_number - 1]}

                    cluster_info.append(entry_info)
                    break
        clusters_info.append(cluster_info)
    text_info['coreferences']['clusters_info'] = clusters_info
    for sentence in text_info['sentences_info']:
        sentence['coreferences'] = {}
    for cluster_number, cluster in enumerate(clusters_info):
        for entry in cluster:
            text_info['sentences_info'][entry['sentence_number']][
                'coreferences'][entry['symbols_spans']] = entry
            text_info['sentences_info'][entry['sentence_number']][
                'coreferences'][entry['symbols_spans']]['cluster'] = cluster_number


def example_usage():
    # text_info
    text_info = get_text_info()


    # create coreferences clusters
    text_info_with_coreferences_clusters = get_text_info()
    create_coreferences_clusters(text_info_with_coreferences_clusters)
    add_semantic_roles(text_info_with_coreferences_clusters)

    # semantic_roles with coreferences
    print("DONE")


if __name__ == '__main__':
    example_usage()
