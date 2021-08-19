from enum import Enum
from typing import Dict

from mapcore.swm.src.components.semnet import Sign
from pyvis.network import Network

from src.script_extraction.sign.extract_script import create_signs
from src.script_extraction.text_preprocessing.words_object import Roles
from src.text_info_restaurant import create_text_info_restaurant

from nltk.corpus import wordnet as wn


class SignNode(Enum):
    color = '#0047AB'
    size = 15


class SignifincanceNode(Enum):
    color = '#6495ED'
    size = 13


class RoleNode(Enum):
    color = '#8FD14F'
    size = 13


def get_definition(lemma: str, synset_number: int):
    # all possible synsets
    synsets = wn.synsets(lemma)

    if not len(synsets):
        return "Unique"

    # concrete synset
    ss = synsets[synset_number]
    return ss.definition()


def show_script(script: Dict[str, Sign], group_roles: bool = False, save_to_file: bool = False):
    net = Network(height='100%', width='100%', notebook=save_to_file, directed=False)

    # All possible roles
    role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}
    int_role: Dict[int, Roles] = {i: role for i, role in enumerate(Roles)}

    # Create all sign nodes
    for sign in script.values():
        net.add_node(n_id=sign.name,
                     color=SignNode.color.value,
                     size=SignNode.size.value)

    # Create all existing definition nodes and connect them to other definitions
    for sign in script.values():
        for cm_index, cm in sign.significances.items():
            name = f'{sign.name}:{cm_index}'
            # existing definitions in current sign
            for event_index, event in enumerate(cm.cause):
                if len(event.coincidences):
                    net.add_node(n_id=name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    net.add_edge(source=sign.name,
                                 to=name,
                                 title=get_definition(lemma=sign.name,
                                                      synset_number=cm_index - 1))
                    if group_roles:
                        net.add_node(n_id=f'{sign.name}:{int_role[event_index].value}',
                                     color=RoleNode.color.value,
                                     size=RoleNode.size.value)
                        net.add_edge(source=name,
                                     to=f'{sign.name}:{int_role[event_index].value}',
                                     title=int_role[event_index].value)

                    # link to definition of role
                    for connector in event.coincidences:
                        out_name = f'{connector.out_sign.name}:{connector.out_index}'
                        net.add_node(n_id=out_name,
                                     color=SignifincanceNode.color.value,
                                     size=SignifincanceNode.size.value)
                        if not group_roles:
                            net.add_edge(source=name,
                                         to=out_name,
                                         label=int_role[event_index].value,
                                         color='#8FD14F' )
                        else:
                            net.add_edge(source=f'{sign.name}:{int_role[event_index].value}',
                                         to=out_name,
                                         )
                        net.add_edge(source=connector.out_sign.name,
                                     to=out_name,
                                     title=get_definition(lemma=connector.out_sign.name,
                                                          synset_number=connector.out_index - 1))

        # images
        for cm_index, cm in sign.images.items():
            name = f'{sign.name}:{cm_index}'
            for event_index, event in enumerate(cm.cause):
                if len(event.coincidences):
                    net.add_node(n_id=name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    net.add_edge(source=sign.name,
                                 to=name,
                                 title=get_definition(lemma=sign.name,
                                                      synset_number=cm_index - 1))
                for connector in event.coincidences:
                    out_name = f'{connector.out_sign.name}:{connector.out_index}'
                    net.add_node(n_id=out_name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    net.add_edge(source=name,
                                 to=out_name,
                                 color="#808080")
                    net.add_edge(source=connector.out_sign.name,
                                 to=out_name,
                                 title=get_definition(lemma=connector.out_sign.name,
                                                      synset_number=connector.out_index - 1))
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
         "gravitationalConstant": -10050
          }
      }
    }
    """)
    net.show("Script.html")


def main():
    text_info = create_text_info_restaurant()
    script = create_signs(text_info)

    show_script(script, group_roles=True)

    print("DONE")


if __name__ == '__main__':
    main()
