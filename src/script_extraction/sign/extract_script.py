"""
Create script from text info
"""
import enum
from typing import Any, Dict, List, Union, Iterable, Optional, Tuple, Set

from mapcore.swm.src.components.semnet import Sign, Event, CausalMatrix, Connector

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, \
    combine_actions_with_clusters
from src.script_extraction.text_preprocessing.words_object import Action, Cluster, WordsObject, Obj, Roles
from src.text_info_restaurant import create_text_info_restaurant
from itertools import combinations


def create_sign(obj: Union[WordsObject, Obj, Action],
                max_roles: int) -> Optional[Sign]:
    # not possible to add links to if no len
    if obj.synsets_len == -1:
        return None
    name = obj.lemma

    sign = Sign(name)

    # add meanings and significances for each wn meaning
    for i in range(obj.synsets_len):
        # Creating place for adding images
        image = sign.add_image(pm=None)
        image.add_event(event=Event(order=0))

        # add signifincances
        significance = sign.add_significance(pm=None)

        if isinstance(obj, Action):
            for i in range(max_roles):
                significance.add_event(event=Event(order=i))
    return sign


# def add_new_sign(obj: Union[WordsObject, Obj, Action],
#                  max_roles: int) -> Sign:
#     name = obj.lemma
#
#     if name not in script:
#         sign = Sign(name)
#
#         # add meanings and significances for each wn meaning
#         for i in range(obj.synsets_len):
#             image = sign.add_image(pm=None)
#             image.add_event(event=Event(order=0))
#             significance = sign.add_significance(pm=None)
#
#             # add events to action for each role
#             if isinstance(obj, Action):
#                 for i in range(max_roles):
#                     significance.add_event(event=Event(order=i))
#
#         script[name] = sign
#     else:
#         # if action sign already created and has no roles
#         if isinstance(obj, Action):
#             sign = script[name]
#             if len(sign.significances[1].cause) != obj.synsets_len:
#                 for significance_number in sign.significances:
#                     for i in range(max_roles):
#                         sign.significances[significance_number].add_event(event=Event(order=i))
#     return script[name]


def add_role_object_to_action(action_sign: Sign,
                              action: Action,
                              role_sign: Sign,
                              role_object: Obj,
                              roles_dict: Dict[Roles, int],
                              arg_type: Optional[Roles] = None) -> None:
    if role_object.synsets_len == -1:
        return
    if arg_type is None:
        arg_type = role_object.arg_type
    action_cm: CausalMatrix = action_sign.significances[action.synset_number + 1]
    # action_cm.add_feature(script[role_object.lemma].significances[role_object.synset_number + 1])
    action_event: Event = action_cm.cause[roles_dict[arg_type]]
    connector: Connector = Connector(in_sign=action_sign,
                                     out_sign=role_sign,
                                     in_index=action.synset_number + 1,  # action signifincance number
                                     out_index=role_object.synset_number + 1,
                                     # role significance number
                                     in_order=roles_dict[
                                         arg_type])  # role number (Event number)
    action_event.add_coincident(base='significance', connector=connector)


def add_image_to_object(object_sign: Sign,
                        obj: Obj,
                        image_sign: Sign,
                        image: WordsObject) -> None:
    if image.synsets_len != -1:
        obj_image_matrix: CausalMatrix = object_sign.images[obj.synset_number + 1]
        obj_image_event: Event = obj_image_matrix.cause[0]

        # connector to image
        connector: Connector = Connector(in_sign=object_sign,
                                         out_sign=image_sign,
                                         in_index=obj.synset_number + 1,  # obj images number
                                         out_index=image.synset_number + 1,  # image images number
                                         in_order=0)
        obj_image_event.add_coincident(base='image', connector=connector)


def add_action_sign_to_script(script: Sign, action_cm: CausalMatrix) -> None:
    cm: CausalMatrix = script.significances[1]
    cm.add_feature(action_cm)


def create_signs(text_info: Dict[str, Any]) -> Tuple[List[Sign], Dict[str, Sign]]:
    # support signs
    actions_signs: List[Sign] = []
    objects_signs: Dict[str, Sign] = dict()

    # Information preparation
    actions: List[Action] = extract_actions(text_info)
    clusters: List[Cluster] = extract_clusters(text_info)
    combine_actions_with_clusters(actions, clusters, text_info)

    # All possible roles
    roles_dict = {role: i for i, role in enumerate(Roles)}

    # Add signs to script
    for action in actions:
        # if no roles, look next action
        if not len(action.objects):
            actions_signs.append(None)
            continue
        action_sign: Sign = create_sign(obj=action, max_roles=len(roles_dict))
        # add_action_sign_to_script(script, action_sign.significances[action.synset_number + 1])
        actions_signs.append(action_sign)

        for obj in action.objects:
            # if no wordnet meanings -> no signifincances matrices ->
            # we can't add connection to this Sign -> Look next
            if obj.lemma not in objects_signs:
                objects_signs[obj.lemma] = create_sign(obj=obj, max_roles=len(roles_dict))
            for image in obj.images:
                if image.lemma not in objects_signs:
                    objects_signs[image.lemma] = create_sign(obj=image, max_roles=len(roles_dict))

    # add role-fillers to actions
    for action_index, action in enumerate(actions):
        if not len(action.objects):
            continue
        action_sign = actions_signs[action_index]
        for role_object in action.objects:
            if role_object.cluster is None:
                add_role_object_to_action(action_sign=action_sign,
                                          action=action,
                                          role_sign=objects_signs[role_object.lemma],
                                          role_object=role_object,
                                          roles_dict=roles_dict)

            else:
                for role_cluster_object in role_object.cluster.real_objects:
                    add_role_object_to_action(action_sign=action_sign,
                                              action=action,
                                              role_sign=objects_signs[role_cluster_object.lemma],
                                              role_object=role_cluster_object,
                                              roles_dict=roles_dict,
                                              arg_type=role_object.arg_type)

    # add images
    for action in actions:
        for obj in action.objects:
            if obj.cluster is None:
                for image in obj.images:
                    add_image_to_object(objects_signs[obj.lemma], obj,
                                        objects_signs[image.lemma], image)
            else:
                for image in obj.cluster.images.values():
                    add_image_to_object(objects_signs[obj.lemma], obj,
                                        objects_signs[image.lemma], image)

    # clear Nones
    actions_signs = [sign for sign in actions_signs if sign is not None]

    return actions_signs, objects_signs


def find_connected_pairs(actions_signs: List[Sign],
                         objects_signs: Dict[str, Sign]) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for object_sign in objects_signs.values():
        if object_sign is None:
            continue
        in_signs: List[Sign] = []
        for connector in object_sign.out_significances:
            already_in: bool = False
            for sign in in_signs:
                if connector.in_sign is sign:
                    already_in = True
                break
            if not already_in:
                in_signs.append(connector.in_sign)
        in_numbers = []
        for in_sign in in_signs:
            for i, act_sign in enumerate(actions_signs):
                if act_sign is in_sign:
                    in_numbers.append(i)
        for elem in combinations(in_numbers, 2):
            pairs.add(elem)
    return pairs


def extract_script(actions_signs: List[Sign],
                   objects_signs: Dict[str, Sign]) -> Sign:
    script = Sign("script")

    pairs = find_connected_pairs(actions_signs, objects_signs)



    print("Done")

    return script


def main():
    text_info = create_text_info_restaurant()

    actions_signs, objects_signs = create_signs(text_info)
    extract_script(actions_signs, objects_signs)
    print("DONE")


if __name__ == '__main__':
    main()
