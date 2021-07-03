class graph:
    __nodes_number = 0

    def __init__(self, text_info):
        graph.__nodes_number = 0
        self.trees = []
        for sentence_info in text_info['sentences_info']:
            self.trees.append(graph.get_tree(sentence_info))
        self.G = graph.get_graph_from_trees(trees=self.trees)

    @staticmethod
    def get_graph_from_trees(trees):
        V = {}
        E = {}

        # removing empty sentences
        trees = [tree for tree in trees if len(tree['children'])]

        # if all text has no verbs return empty graph
        if not len(trees):
            return V, E

        # vertices from last step to create links
        start_vertex = graph.get_start_vertice()
        current_vertices = {start_vertex['node_number']: start_vertex}
        V.update(current_vertices)

        # iterating through all roots from each tree
        for tree_number, tree in enumerate(trees):
            # here will be added all roots nodes from current sentence
            # to link them to next sentence
            new_current_vertices = {}

            # iterating through all children
            for child_number, child in enumerate(tree['children']):
                # getting vertex
                v = graph.get_vertice_from_dict(child)
                node_number = v['node_number']

                # add created vertex to dict
                new_current_vertices[node_number] = v

            # creating links from previous vertex to all next
            for node_number in current_vertices:
                E[node_number] = list(new_current_vertices.keys())

            # next step, switching roots to link
            current_vertices = new_current_vertices
            V.update(current_vertices)

        # add end vertex
        end_vertex = graph.get_end_vertice()

        return V, E

    @staticmethod
    def get_vertice_from_dict(v_dict):
        v = {}
        for key in v_dict:
            if key != 'children':
                v[key] = v_dict[key]
        return v

    @staticmethod
    def fill_V_E_from_trees(V, E):
        pass

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

    @staticmethod
    def get_start_vertice():
        start = {'node_number': -1,
                 'word': 'start'}
        return start

    @staticmethod
    def get_end_vertice():
        end = {'node_number': -2,
               'word': 'end'}
        return end


def example_usage():
    # get_tree
    from tests.get_info import get_sentence_info
    sentence_info = get_sentence_info()
    root = graph.get_tree(sentence_info)

    from tests.get_info import get_text_info
    text_info = get_text_info()
    G = graph(text_info)

    from tests.get_info import get_text_info
    _text_info = get_text_info()
    _G = graph(text_info)
    start_vertice = _G.get_start_vertice()
    end_vertice = _G.get_end_vertice()
    del _G, _text_info

    from tests.get_info import get_text_info
    _text_info = get_text_info()
    _G = graph(text_info)
    vertice_from_dict = _G.get_vertice_from_dict(_G.trees[0]['children'][0])
    del _G, _text_info

    from tests.get_info import get_text_info
    _text_info = get_text_info()
    _G = graph(text_info)
    V, E = _G.get_graph_from_trees(_G.trees)
    del _G, _text_info
    print("Done")


if __name__ == '__main__':
    example_usage()
