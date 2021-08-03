from typing import Dict, Any, List, Set

from mapcore.swm.src.components.semnet import Sign, Connector
from mapcore.swm.src.components.semnet import CausalMatrix
from pyvis.network import Network

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, Action, Obj, Roles, Image
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from src.script_extraction.visualization.show_script_graph import show_script_graph


def add_image_sign(image: Image,
                   obj_meaning: CausalMatrix,
                   images_signs: Dict[str, Sign]):
    image_name = image.lemma()
    if image_name not in images_signs:
        images_signs[image_name] = Sign(image_name)
        images_signs[image_name].add_significance()
    image_signinficances = images_signs[image_name].significances[1]
    connector = obj_meaning.add_feature(image_signinficances)
    images_signs[image_name].add_out_meaning(connector)


def add_obj_sign(obj: Obj,
                 action_significance: CausalMatrix,
                 objects_signs: Dict[str, Sign],
                 roles_signs: Dict[str, Sign],
                 images_signs: Dict[str, Sign]):
    # add role
    role_significance: CausalMatrix = roles_signs[obj.arg_type.value].add_significance()
    connector: Connector = action_significance.add_feature(role_significance, zero_out=False)
    roles_signs[obj.arg_type.value].add_out_significance(connector)

    # add object
    obj_name = obj.lemma()
    if obj_name not in objects_signs:
        objects_signs[obj_name] = Sign(obj_name)
        objects_signs[obj_name].add_significance()
        objects_signs[obj_name].add_meaning()
    obj_significance = objects_signs[obj_name].significances[1]
    connector = role_significance.add_feature(obj_significance, zero_out=True)
    objects_signs[obj_name].add_out_significance(connector)

    for image in obj.images:
        add_image_sign(image, objects_signs[obj_name].meanings[1], images_signs)


def add_action_sign(action: Action,
                    S: Sign,
                    actions_signs: Dict[str, Sign],
                    significances: Dict[str, CausalMatrix],
                    objects_signs: Dict[str, Sign],
                    roles_signs: Dict[str, Sign],
                    images_signs: Dict[str, Sign]):
    action_name = action.lemma()

    if action_name not in actions_signs:
        actions_signs[action_name] = Sign(action_name)
    action_significance: CausalMatrix = actions_signs[action_name].add_significance()
    connector: Connector = S.significances[1].add_feature(action_significance, order=None, zero_out=False)
    actions_signs[action_name].add_out_significance(connector)

    for obj in action.objects:
        add_obj_sign(obj, action_significance, objects_signs, roles_signs, images_signs)

    for child_act in action.actions:
        add_action_sign(action=child_act,
                        S=S,
                        actions_signs=actions_signs,
                        significances=significances,
                        objects_signs=objects_signs,
                        roles_signs=roles_signs,
                        images_signs=images_signs)


def add_role_sign(role, roles_signs):
    roles_signs[role] = Sign(role)


def create_script_sign(text_info: Dict[str, Any]):
    script_name = "Script"
    S = Sign(script_name)

    actions_signs: Dict[str, Sign] = {}
    roles_signs: Dict[str, Sign] = {}
    objects_signs: Dict[str, Sign] = {}
    images_signs: Dict[str, Sign] = {}

    significances: Dict[str, CausalMatrix] = {}
    significances[script_name] = S.add_significance()

    for role in Roles:
        add_role_sign(role.value, roles_signs)
    actions = extract_actions(text_info)
    for action in actions:
        add_action_sign(action=action,
                        S=S,
                        actions_signs=actions_signs,
                        significances=significances,
                        objects_signs=objects_signs,
                        roles_signs=roles_signs,
                        images_signs=images_signs)

    show_script_graph(S, actions_signs, objects_signs, roles_signs, images_signs)
    # show_graph(actions)

    print("Done")




def example_usage():
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    create_script_sign(text_info)
    print("Done")


if __name__ == '__main__':
    example_usage()
