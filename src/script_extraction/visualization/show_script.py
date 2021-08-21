from enum import Enum
from typing import Dict, Optional

from mapcore.swm.src.components.semnet import Sign
from pyvis.network import Network

from src.script_extraction.sign.extract_script import create_signs, extract_script
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from src.script_extraction.text_preprocessing.words_object import Roles
from src.text_info_restaurant import create_text_info_restaurant

from nltk.corpus import wordnet as wn


class ScriptNode(Enum):
    name = 'Script'
    color = '#F24726'
    size = 20


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


def display_sign(net: Network,
                 sign: Sign,
                 int_role: Dict[int, Roles],
                 step: Optional[int] = None):
    # for each wn meaning
    for cm_index, cm in sign.significances.items():
        # meaning sub node
        sub_meaning_name = f'{sign.name}:{cm_index}'

        for event_index, event in enumerate(cm.cause):
            if len(event.coincidences):
                # if has connections (roles)
                # create sub meaning node
                # do
                net.add_node(n_id=sub_meaning_name,
                             color=SignifincanceNode.color.value,
                             size=SignifincanceNode.size.value)
                # do -> do:1
                net.add_edge(source=sign.name,
                             to=sub_meaning_name,
                             title=get_definition(lemma=sign.name,
                                                  synset_number=cm_index - 1))

                # create role
                role_name = f'{sign.name}:{cm_index}:{int_role[event_index].value}'
                net.add_node(n_id=role_name,
                             color=RoleNode.color.value,
                             size=RoleNode.size.value)
                # do:1 -> do:1:ARG0
                net.add_edge(source=sub_meaning_name,
                             to=role_name,
                             title=int_role[event_index].value)

                # link to fillers of role
                for connector in event.coincidences:
                    out_sub_node_name = f'{connector.out_sign.name}:{connector.out_index}'
                    # create role filler
                    net.add_node(n_id=out_sub_node_name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    # do:1:ARG0 -> something:1
                    net.add_edge(source=role_name,
                                 to=out_sub_node_name,
                                 label=step if step is not None else "")
                    # something:1 -> something
                    net.add_edge(source=connector.out_sign.name,
                                 to=out_sub_node_name,
                                 title=get_definition(lemma=connector.out_sign.name,
                                                      synset_number=connector.out_index - 1))

        # images
        for cm_index, cm in sign.images.items():
            sub_meaning_name = f'{sign.name}:{cm_index}'
            for event_index, event in enumerate(cm.cause):
                if len(event.coincidences):
                    # create sub_meaning_node
                    net.add_node(n_id=sub_meaning_name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    # do -> do:1
                    net.add_edge(source=sign.name,
                                 to=sub_meaning_name,
                                 title=get_definition(lemma=sign.name,
                                                      synset_number=cm_index - 1))
                for connector in event.coincidences:
                    out_sub_node_name = f'{connector.out_sign.name}:{connector.out_index}'
                    # create out sub meaning node
                    net.add_node(n_id=out_sub_node_name,
                                 color=SignifincanceNode.color.value,
                                 size=SignifincanceNode.size.value)
                    # do:1 -> something:1
                    net.add_edge(source=sub_meaning_name,
                                 to=out_sub_node_name,
                                 label='img',
                                 color="#808080")
                    # something:1 -> something
                    net.add_edge(source=connector.out_sign.name,
                                 to=out_sub_node_name,
                                 title=get_definition(lemma=connector.out_sign.name,
                                                      synset_number=connector.out_index - 1))


def show_script(script: Sign,
                objects_signs: Dict[str, Sign],
                save_to_file: bool = False,
                ):
    remove_list = ["i", "good", "good:1", "love", "mother", "musical", "favorite", "favorite:1", "tradition:1",
                   "tradition", "film", "film:1", "popular:1", "popular", "pleasure", "once", "cinema", "father", "family"]
    net = Network(height='100%', width='100%', notebook=save_to_file, directed=False)

    # All possible roles
    role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}
    int_role: Dict[int, Roles] = {i: role for i, role in enumerate(Roles)}

    net.add_node(n_id=ScriptNode.name.value,
                 color=ScriptNode.color.value,
                 size=ScriptNode.size.value)

    # Create all sign nodes
    for i, cm in enumerate(script.significances.values()):
        for j, event in enumerate(cm.cause):
            sign = list(event.coincidences)[0].out_sign
            net.add_node(n_id=sign.name,
                         color=SignNode.color.value,
                         size=25)
            for edge in net.edges:
                if edge['to'] == sign.name:
                    edge['label'] += f',  {j}'
                    break
            net.add_edge(source=ScriptNode.name.value,
                         to=sign.name,
                         label=f'{j}')

    for sign in objects_signs.values():
        if sign is not None:
            net.add_node(n_id=sign.name,
                         color=SignNode.color.value,
                         size=SignNode.size.value)

    # Create all existing definition nodes and connect them to other definitions
    for i, cm in script.significances.items():
        for j, event in enumerate(cm.cause):
            sign = list(event.coincidences)[0].out_sign
            display_sign(net=net, sign=sign, int_role=int_role, step=f'{j}')

    for sign in objects_signs.values():
        if sign is not None:
            display_sign(net=net, sign=sign, int_role=int_role)

    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
         "gravitationalConstant": -10050
          }
      }
    }
    """)
    for node in net.nodes:
        if node['id'] in remove_list:
            del node
    net.show("Script.html")


def main():
    path = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    # path_john = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/john.txt'
    files = [path]
    text_info = extract_texts_info(files)[0]
    # _text_info = create_text_info_restaurant()

    actions_signs, objects_signs = create_signs(text_info)
    script = extract_script(actions_signs, objects_signs, limit=1)

    show_script(script, objects_signs, save_to_file=False)

    print("DONE")


if __name__ == '__main__':
    main()
