from enum import Enum
from typing import Optional, Dict

import nltk.corpus
from mapcore.swm.src.components.semnet import Sign, Connector
from pyvis.network import Network

from ..script import Script
from ..preprocessing.words_object import Roles
from ..samples.text_info.text_info_restaurant import create_text_info_restaurant


class ScriptNode(Enum):
    name = 'Script'
    color = '#F24726'
    size = 20


class ScriptStepNode(Enum):
    color = '#0047AB'
    size = 23


class SignNode(Enum):
    color = '#0047AB'
    size = 15


class SignifincanceNode(Enum):
    color = '#6495ED'
    size = 13


class RoleNode(Enum):
    color = '#8FD14F'
    size = 13


class ImageEdge(Enum):
    color = "#808080"


class Visualizator:
    def __init__(self, script: Script,
                 directed: Optional[bool] = False,
                 save_to_file: Optional[bool] = False,
                 show_buttons: bool = False,
                 name: Optional[str] = None):


        if name is None:
            name = 'Script'
        name += '.html'
        self.name = name
        # script
        self.script: Sign = script.sign
        self.objects_signs = script.objects_signs

        # manual graph tuning
        self.show_buttons = show_buttons

        # graph
        scale = '50%' if self.show_buttons else '100%'
        self.net = Network(height=scale,
                           width=scale,
                           directed=directed,
                           notebook=save_to_file)

        if not self.show_buttons:
            self.set_physic_options()

        # All possible roles
        self.role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}
        self.int_role: Dict[int, Roles] = {i: role for i, role in enumerate(Roles)}

        # add objects to self.net
        self.create_script_node()
        self.create_script_step_nodes()
        self.create_objects_nodes()
        self.create_objects_edges()


    def show(self):
        if self.show_buttons:
            self.net.show_buttons()

        self.net.show(self.name)
        return self

    def create_script_step_nodes(self) -> None:
        """
        Create all script step nodes
        Add edges and description
        """

        # for each separate script
        for i, significance in self.script.significances.items():
            # for each script step
            for j, event in enumerate(significance.cause):
                script_step_name = f'{ScriptNode.name.value}:{i}'

                # only one coincident - script step
                sign = list(event.coincidences)[0].out_sign

                # create script step node: script_step
                self.net.add_node(n_id=sign.name,
                                  color=ScriptStepNode.color.value,
                                  size=ScriptStepNode.size.value)

                # if edge already exist, add description
                edge_exists = False
                for edge in self.net.edges:
                    if edge['to'] == sign.name and edge['from'] == script_step_name:
                        edge['label'] += f', {i}:{j}'
                        edge_exists = True
                        break

                # create edge if not exist: script ---0:1---> script_step
                if not edge_exists:
                    self.net.add_edge(source=script_step_name,
                                      to=sign.name,
                                      label=f'{i}:{j}')

    def create_script_node(self):
        """
        Add script node and script parts nodes
        """
        self.net.add_node(n_id=ScriptNode.name.value,
                          color=ScriptNode.color.value,
                          size=ScriptNode.size.value)
        for i, significance in self.script.significances.items():
            name = f'{ScriptNode.name.value}:{i}'
            self.net.add_node(n_id=name,
                              color=ScriptNode.color.value,
                              size=ScriptNode.size.value - 3)
            self.net.add_edge(source=ScriptNode.name.value,
                              to=name,
                              label=f'{len(significance.cause)} steps')

    def create_objects_nodes(self):
        for sign in self.objects_signs.values():
            self.net.add_node(n_id=sign.name,
                              color=SignNode.color.value,
                              size=SignNode.size.value)

    def set_physic_options(self) -> None:
        self.net.set_options("""
            var options = {
              "nodes": {
                "borderWidth": 4,
                "borderWidthSelected": 6,
                "font": {
                  "size": 20,
                  "face": "verdana"
                }
              },
              "edges": {
                "color": {
                  "inherit": true
                },
                "font": {
                  "face": "verdana"
                },
                "smooth": false
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -10900,
                  "springLength": 75
                },
                "minVelocity": 0.75
              }
            }      
            """)

    def _process_connector(self, connector: Connector,
                           image: Optional[bool] = False):
        # names
        out_name: str = connector.out_sign.name
        in_name: str = connector.in_sign.name
        out_sub_meaning_name = f'{out_name}:{connector.out_index}'
        in_sub_meaning_name = f'{in_name}:{connector.in_index}'

        #  object:1 node
        self.net.add_node(n_id=out_sub_meaning_name,
                          color=SignifincanceNode.color.value,
                          size=SignifincanceNode.size.value)
        # action:1 node
        self.net.add_node(n_id=in_sub_meaning_name,
                          color=SignifincanceNode.color.value,
                          size=SignifincanceNode.size.value)

        # object -> object:1
        self.net.add_edge(source=out_name,
                          to=out_sub_meaning_name,
                          title=self.get_definition(lemma=out_name,
                                                    synset_number=connector.out_index - 1))
        # action -> action:1
        self.net.add_edge(source=in_name,
                          to=in_sub_meaning_name,
                          title=self.get_definition(lemma=in_name,
                                                    synset_number=connector.in_index - 1))
        # action:1 -> object:1
        self.net.add_edge(source=in_sub_meaning_name,
                          to=out_sub_meaning_name,
                          label=self.int_role[connector.in_order].value if not image else "image",
                          color=RoleNode.color.value if not image else ImageEdge.color.value)

    def create_objects_edges(self):
        for object_sign in self.objects_signs.values():
            for connector in object_sign.out_significances:
                self._process_connector(connector, image=False)
            for connector in object_sign.out_images:
                self._process_connector(connector, image=True)

    @staticmethod
    def get_definition(lemma: str, synset_number: int):
        # all possible synsets
        synsets = nltk.corpus.wordnet.synsets(lemma)

        if not len(synsets):
            return "Unique"

        # concrete synset
        ss = synsets[synset_number]
        return ss.definition()


def example_usage():
    text_info = create_text_info_restaurant()

    script = Script(text_info)

    visualizator = Visualizator(script).show()

    print("Done")


if __name__ == '__main__':
    example_usage()
