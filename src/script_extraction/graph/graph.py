from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from typing import Tuple, Dict, Any, Set, Union
from nltk.corpus import wordnet as wn # type: ignore
from pyvis.network import Network # type: ignore


class Vertex:
    def __init__(self, index: Any, # unique value
                 label: str, # show label for pyvis
                 tp: Union[str, None], # for select color
                 data: Set[str], # strings, representing of possible values for role
                 pos: str = "", # part of speech for adding synonyms... etc
                 ) -> None:
        self.index = index
        self.label = label
        self.tp = tp
        self.data = data
        self.pos = pos

    def __repr__(self):
        return f'{self.index}:{self.pos}:{self.label}'


class Edge:
    def __init__(self, v1: Tuple[int, Tuple[int, int]], v2,
                 label: str = ""):
        self.v1 = v1
        self.v2 = v2
        self.label = label

    def index(self):
        return self.v1, self.v2

    def __repr__(self):
        return f"{self.index().__repr__()}:{self.label}"


class Graph:
    def __init__(self, text_info):
        # extract clusters
        self.clusters, self.elements_dict = extract_clusters(text_info)
        self.parts_of_speech = [sentence_info['dependency']['pos'] for sentence_info in text_info['sentences_info']]
        # extract roles
        self.verbs = []
        for sentence_number, sentence_info in enumerate(text_info['sentences_info']):
            self.verbs += extract_actions(sentence_info, sentence_number)

        self.V: Dict[Any, Vertex] = dict()
        self.E: Dict[Any, Edge] = dict()

        for verb in self.verbs:
            self.add_verb_to_graph(verb)

    def get_graph(self) -> Tuple[Dict[Any, Vertex], Dict[Any, Edge]]:
        return self.V, self.E

    def add_verb_to_graph(self, verb: Role):
        # if no childs
        if not len(verb.objects):
            return None
        parent_v = self.add_vertex(verb)

        for role in verb.objects:
            child_v = self.add_vertex(role)
            self.add_edge(parent_v.index, child_v.index, role.argument_type)

    def add_vertex(self, role: Role) -> Vertex:
        if role.index() in self.V:
            return self.V[role.index()]
        # create vertex if not exist
        else:
            role_start_word_num = role.index()[1][0]
            role_end_word_num = role.index()[1][1]
            role_sentence = role.index()[0]
            articles = ['the', 'a', 'an']

            part_of_speech = ""
            if role_end_word_num - role_start_word_num == 1:
                part_of_speech = self.parts_of_speech[role_sentence][role_start_word_num]
            elif (role_end_word_num - role_start_word_num == 2
                  and role.text.split()[0].lower() in articles):
                    part_of_speech = self.parts_of_speech[role_sentence][role_start_word_num + 1]
                    role.text = role.text.split()[1]

            # checking vertex for being in cluster
            if role.index() in self.elements_dict:
                cluster = self.elements_dict[role.index()]

                if cluster not in self.V:
                    cluster_vertex = Vertex(index=cluster,
                                            label=str(self.clusters[cluster].get_elements()),
                                            tp="cluster",
                                            pos=part_of_speech,
                                            data=self.clusters[cluster].get_elements())
                    self.V[cluster] = cluster_vertex

                return self.V[cluster]
            else:
                # create vertex
                vertex = Vertex(index=role.index(),
                                label=role.text,
                                data={role.text},
                                pos=part_of_speech,
                                tp=None)
                self.V[role.index()] = vertex
                return vertex

    def add_edge(self, v1, v2, label=""):
        edge = Edge(v1, v2, label)
        self.E[edge.index()] = edge


def show_graph(V: Dict[Any, Vertex], E, path):
    # create pyvis Network
    net = Network(notebook=True, height='100%', width='100%')

    # add nodes
    for v in V:
        color = None
        if V[v].tp == 'cluster':
            color = '#FA7E1E'
        net.add_node(str(V[v].index), label=V[v].label, color=color, size=10, title=V[v].data.__repr__())
    # add edges
    for e in E:
        net.add_edge(str(E[e].v1), str(E[e].v2), label=E[e].label)
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
         "gravitationalConstant": -10050
          }
      }
    }
    """)
    net.show(path)


def add_hyponyms_hypernyms_synonyms(V: Dict[Any, Vertex],
                                    synonyms: bool = False,
                                    hyponyms: bool = False,
                                    hypernyms: bool = False) -> None:
    allowed_pos_types = {'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
    # iterate by vertices
    for v in V:
        # if we require this part of speech
        if (V[v].pos is not None) and (V[v].pos in allowed_pos_types):

            extend_data = set()
            # checking each word in vertex
            for word in V[v].data:
                # checking each definition of this word
                for ss in wn.synsets(word):
                    # if part of speech matches
                    if ss.pos() == allowed_pos_types[V[v].pos]:
                        # update our set
                        if synonyms:
                            extend_data.update(ss.lemma_names())
                        if hypernyms:
                            for hypernym_ss in ss.hypernyms():
                                extend_data.update(hypernym_ss.lemma_names())
                        if hyponyms:
                            for hyponym_ss in ss.hyponyms():
                                extend_data.update(hyponym_ss.lemma_names())
            # add data to vertex
            V[v].data.update(extend_data)
    return None


def example_usage():
    text_info = get_text_info()

    graph = Graph(text_info)
    graph.show_graph('example.html')


if __name__ == '__main__':
    example_usage()
