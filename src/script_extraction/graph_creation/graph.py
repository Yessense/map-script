class graph:
    __nodes_number = 0

    def __init__(self, text_info):
        self.trees = []
        for sentence_info in text_info['sentences_info']:
            self.trees.append(graph.get_tree(sentence_info))

    @staticmethod
    def get_tree(sentence_info):
        """
        Parameters
        ----------
        sentence_info

        Returns
        -------
        out: tree
            verb Tree
        """
        hierplane_tree = sentence_info['dependency']['hierplane_tree']
        root = {'children': [],
                'nodeType': 'tree_root',
                'node_number': graph.__nodes_number}
        graph.__nodes_number += 1
        graph.recursive_verb_search(hierplane_tree['root'], root)
        return root

    @staticmethod
    def add_child(node_for_adding, node):
        """
        add child to node in node['children'] list

        Parameters
        ----------
        node_for_adding: dict
        node: dict

        Returns
        -------
        dict
        """

        # add empty dict
        node_for_adding['children'].append({})

        # add all values except children, cause we need only verbs
        for key, value in node.items():
            if key != 'children':
                node_for_adding['children'][-1][key] = value

        # add empty children list to avoid None value reference in dict
        node_for_adding['children'][-1]['children'] = []

        # add unique index
        node_for_adding['children'][-1]['node_number'] = graph.__nodes_number
        graph.__nodes_number += 1

        return node_for_adding['children'][-1]

    @staticmethod
    def recursive_verb_search(node, node_for_adding):
        if 'VERB' in node['attributes']:
            # current word is a verb
            # add him to tree
            # next nodes will be added to him
            added_node = graph.add_child(node_for_adding, node)
            if 'children' in node:
                for child in node['children']:
                    graph.recursive_verb_search(child, added_node)
        else:
            if 'children' in node:
                # checking children for verbs
                for child in node['children']:
                    graph.recursive_verb_search(child, node_for_adding)
            else:
                # current word is not a verb and there is no children
                return


def example_usage():
    # get_tree
    sentence_info = {
        'dependency': {'arc_loss': 0.2748850882053375, 'tag_loss': 0.285218209028244, 'loss': 0.5601032972335815,
                       'words': ['This', 'week', 'we', 'decided', 'to', 'look', 'at', 'a', 'small', 'family', '-',
                                 'run', 'restaurant', 'in', 'the', 'village', 'of', 'Wardleton', '.'],
                       'pos': ['DET', 'NOUN', 'PRON', 'VERB', 'PART', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT',
                               'VERB', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'PROPN', 'PUNCT'],
                       'predicted_dependencies': ['nsubj', 'dep', 'nsubj', 'root', 'dep', 'xcomp', 'prep', 'dep', 'dep',
                                                  'dep', 'dep', 'dep', 'dep', 'prep', 'dep', 'pobj', 'prep', 'pobj',
                                                  'punct'],
                       'predicted_heads': [4, 4, 4, 0, 6, 4, 6, 13, 10, 13, 13, 13, 6, 13, 16, 14, 16, 17, 4],
                       'hierplane_tree': {
                           'text': 'This week we decided to look at a small family - run restaurant in the village of Wardleton .',
                           'root': {'word': 'decided', 'nodeType': 'root', 'attributes': ['VERB'], 'link': 'root',
                                    'spans': [{'start': 13, 'end': 21}], 'children': [
                                   {'word': 'This', 'nodeType': 'nsubj', 'attributes': ['DET'], 'link': 'nsubj',
                                    'spans': [{'start': 0, 'end': 5}]},
                                   {'word': 'week', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep',
                                    'spans': [{'start': 5, 'end': 10}]},
                                   {'word': 'we', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj',
                                    'spans': [{'start': 10, 'end': 13}]},
                                   {'word': 'look', 'nodeType': 'xcomp', 'attributes': ['VERB'], 'link': 'xcomp',
                                    'spans': [{'start': 24, 'end': 29}], 'children': [
                                       {'word': 'to', 'nodeType': 'dep', 'attributes': ['PART'], 'link': 'dep',
                                        'spans': [{'start': 21, 'end': 24}]},
                                       {'word': 'at', 'nodeType': 'prep', 'attributes': ['ADP'], 'link': 'prep',
                                        'spans': [{'start': 29, 'end': 32}]},
                                       {'word': 'restaurant', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep',
                                        'spans': [{'start': 53, 'end': 64}], 'children': [
                                           {'word': 'a', 'nodeType': 'dep', 'attributes': ['DET'], 'link': 'dep',
                                            'spans': [{'start': 32, 'end': 34}]},
                                           {'word': 'family', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep',
                                            'spans': [{'start': 40, 'end': 47}], 'children': [
                                               {'word': 'small', 'nodeType': 'dep', 'attributes': ['ADJ'],
                                                'link': 'dep', 'spans': [{'start': 34, 'end': 40}]}]},
                                           {'word': '-', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep',
                                            'spans': [{'start': 47, 'end': 49}]},
                                           {'word': 'run', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep',
                                            'spans': [{'start': 49, 'end': 53}]},
                                           {'word': 'in', 'nodeType': 'prep', 'attributes': ['ADP'], 'link': 'prep',
                                            'spans': [{'start': 64, 'end': 67}], 'children': [
                                               {'word': 'village', 'nodeType': 'pobj', 'attributes': ['NOUN'],
                                                'link': 'pobj', 'spans': [{'start': 71, 'end': 79}], 'children': [
                                                   {'word': 'the', 'nodeType': 'dep', 'attributes': ['DET'],
                                                    'link': 'dep', 'spans': [{'start': 67, 'end': 71}]},
                                                   {'word': 'of', 'nodeType': 'prep', 'attributes': ['ADP'],
                                                    'link': 'prep', 'spans': [{'start': 79, 'end': 82}], 'children': [
                                                       {'word': 'Wardleton', 'nodeType': 'pobj',
                                                        'attributes': ['PROPN'], 'link': 'pobj',
                                                        'spans': [{'start': 82, 'end': 92}]}]}]}]}]}]},
                                   {'word': '.', 'nodeType': 'punct', 'attributes': ['PUNCT'], 'link': 'punct',
                                    'spans': [{'start': 92, 'end': 94}]}]},
                           'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'],
                                               'nsubj': ['color1'], 'nsubjpass': ['color1'], 'csubj': ['color1'],
                                               'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'],
                                               'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'],
                                               'xcomp': ['color2'], 'ccomp': ['color2'], 'acomp': ['color2'],
                                               'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'],
                                               'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'],
                                               'number': ['color3'], 'possesive': ['color3'], 'poss': ['color3'],
                                               'discourse': ['color3'], 'expletive': ['color3'], 'prt': ['color3'],
                                               'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'],
                                               'tmod': ['color4'], 'quantmod': ['color4'], 'npadvmod': ['color4'],
                                               'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'],
                                               'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']},
                           'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left',
                                              'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right',
                                              'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'}}},
        'semantic role': {'verbs': [{'verb': 'decided',
                                     'description': '[ARGM-TMP: This week] [ARG0: we] [V: decided] [ARG1: to look at a small family - run restaurant in the village of Wardleton] .',
                                     'tags': ['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1',
                                              'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1',
                                              'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']}, {'verb': 'look',
                                                                                              'description': 'This week [ARG0: we] decided to [V: look] [ARG1: at a small family - run restaurant in the village of Wardleton] .',
                                                                                              'tags': ['O', 'O',
                                                                                                       'B-ARG0', 'O',
                                                                                                       'O', 'B-V',
                                                                                                       'B-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1',
                                                                                                       'I-ARG1', 'O']},
                                    {'verb': 'run',
                                     'description': 'This week we decided to look at a small [ARG0: family] - [V: run] [ARG1: restaurant] in the village of Wardleton .',
                                     'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG0', 'O', 'B-V',
                                              'B-ARG1', 'O', 'O', 'O', 'O', 'O', 'O']}],
                          'words': ['This', 'week', 'we', 'decided', 'to', 'look', 'at', 'a', 'small', 'family', '-',
                                    'run', 'restaurant', 'in', 'the', 'village', 'of', 'Wardleton', '.']}}
    root = graph.get_tree(sentence_info)

    # text_info
    # sentences_info =

    print("Done")


if __name__ == '__main__':
    example_usage()
