from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_semantic_roles
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from src.script_extraction.text_preprocessing.role import Role
from tests.get_info import get_text_info
from typing import Tuple, Dict, Any


class Vertex:
    def __init__(self, index: Any, label: str, data, cluster=None):
        self.index = index
        self.label = label
        self.data = data
        self.cluster = cluster

    def __repr__(self):
        return f'{self.index}:{self.label}'


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

        # extract roles
        self.verbs = []
        for sentence_number, sentence_info in enumerate(text_info['sentences_info']):
            self.verbs += extract_semantic_roles(sentence_info, sentence_number)

        self.V: Dict[Any: Vertex] = {}
        self.E: Dict[Any: Edge] = {}

        for verb in self.verbs:
            self.add_verb_to_graph(verb)

    def add_verb_to_graph(self, verb: Role):
        # if no childs
        if not len(verb.roles):
            return None
        self.add_vertex(verb)

        for role in verb.roles:
            self.add_vertex(role)
            self.add_edge(verb.index(), role.index(), role.argument_type)

    def add_vertex(self, role: Role) -> None:
        # create vertex if not exist
        if role.index() not in self.V:
            # checking vertex for being in cluster
            cluster = None
            if role.index() in self.elements_dict:
                cluster = self.elements_dict[role.index()]

                if cluster not in self.V:
                    cluster_vertex = Vertex(index=cluster,
                                            label=str(self.clusters[cluster]),
                                            data=self.clusters[cluster])
                    self.V[cluster] = cluster_vertex

            # create vertex
            vertex = Vertex(index=role.index(),
                            label=role.text,
                            data=role,
                            cluster=cluster)
            self.V[role.index()] = vertex

            if cluster:
                self.add_edge(role.index(), self.V[cluster].index, label=f'Noun phrase:{cluster}')


    def add_edge(self, v1, v2, label=""):
        edge = Edge(v1, v2, label)
        self.E[edge.index()] = edge


def example_usage():
    text_info = get_text_info()

    graph = Graph(text_info)
    V: Dict[Any, Vertex] =  graph.V
    E: Dict[Any, Edge] = graph.E

    from pyvis.network import Network
    net = Network(notebook=True, height='100%', width='100%')

    for v in V:
        color = None
        if V[v].data.argument_type == 'V':
            color = '#2f7ed8'
        if V[v].data.argument_type == 'cluster':
            color = '#FA7E1E'

        net.add_node(str(V[v].index), label=V[v].label, color=color, size=10)
    for e in E:
        net.add_edge(str(E[e].v1), str(E[e].v2), label=E[e].label)
    # net.set_options(options='{"physics": { "barnesHut": { "gravitationalConstant": -10000 }}}'j)
    net.show_buttons()
    net.show("graph.html")
    print("DONE")


if __name__ == '__main__':
    example_usage()
