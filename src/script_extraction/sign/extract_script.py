import enum
from typing import Any, Dict, List, Union, Iterable, Optional

from mapcore.swm.src.components.semnet import Sign, Event, CausalMatrix, Connector

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, \
    combine_actions_with_clusters
from src.script_extraction.text_preprocessing.words_object import Action, Cluster, WordsObject, Obj, Roles
from src.text_info_restaurant import create_text_info_restaurant


def add_new_sign(script: Dict[str, Sign],
                 obj: Union[WordsObject, Obj, Action],
                 max_roles: int) -> Dict[str, Sign]:
    name = obj.lemma

    if name not in script:
        sign = Sign(name)

        # add meanings and significances for each wn meaning
        for i in range(obj.synsets_len):
            image = sign.add_image(pm=None)
            image.add_event(event=Event(order=0))
            significance = sign.add_significance(pm=None)

            # add events to action for each role
            if isinstance(obj, Action):
                for i in range(max_roles):
                    significance.add_event(event=Event(order=i))

        script[name] = sign
    else:
        # if action sign already created and has no roles
        if isinstance(obj, Action):
            sign = script[name]
            if len(sign.significances[1].cause) != obj.synsets_len:
                for significance_number in sign.significances:
                    for i in range(max_roles):
                        sign.significances[significance_number].add_event(event=Event(order=i))

    return script


def add_role_object_to_action(script: Dict[str, Sign],
                              roles_dict: Dict[Roles, int],
                              action: Action,
                              role_object: Obj,
                              arg_type: Optional[Roles] = None) -> None:
    if role_object.synsets_len == -1:
        return
    if arg_type is None:
        arg_type = role_object.arg_type
    action_sign = script[action.lemma]
    role_sign = script[role_object.lemma]
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

def add_image_to_object(script: Dict[str, Sign],
                        obj: Obj,
                        image: WordsObject):
    if image.synsets_len != -1:
        obj_image_matrix: CausalMatrix = script[obj.lemma].images[obj.synset_number + 1]
        obj_image_event: Event = obj_image_matrix.cause[0]

        # connector to image
        connector: Connector = Connector(in_sign=script[obj.lemma],
                                         out_sign=script[image.lemma],
                                         in_index=obj.synset_number + 1,  # obj images number
                                         out_index=image.synset_number + 1,  # image images number
                                         in_order=0)
        obj_image_event.add_coincident(base='image', connector=connector)


def extract_script(text_info: Dict[str, Any]):
    script: Dict[str, Sign] = dict()

    # Information preparation
    actions: List[Action] = extract_actions(text_info)
    clusters: List[Cluster] = extract_clusters(text_info)
    combine_actions_with_clusters(actions, clusters, text_info)

    # All possible roles
    roles_dict = {role: i for i, role in enumerate(Roles)}

    # Add signs to script
    for action in actions:
        if len(action.objects):
            add_new_sign(script=script, obj=action, max_roles=len(roles_dict))
            for obj in action.objects:
                add_new_sign(script=script, obj=obj, max_roles=len(roles_dict))
                if obj.synsets_len != -1:
                    for image in obj.images:
                        add_new_sign(script=script, obj=image, max_roles=len(roles_dict))

    # add role-fillers to actions
    for action in actions:
        if len(action.objects):
            action_sign = script[action.lemma]
            for role_object in action.objects:
                if role_object.cluster is None:
                    add_role_object_to_action(script, roles_dict, action, role_object)
                else:
                    for role_cluster_object in role_object.cluster.real_objects:
                        add_role_object_to_action(script, roles_dict, action, role_cluster_object,
                                                  arg_type=role_object.arg_type)

    # add images
    for action in actions:
        for obj in action.objects:
            if obj.synsets_len == -1:
                continue
            if obj.cluster is None:
                for image in obj.images:
                    add_image_to_object(script, obj, image)
            else:
                for image in obj.cluster.images.values():
                    add_image_to_object(script, obj, image)

    return script


def main():
    text_info = create_text_info_restaurant()

    script = extract_script(text_info)

    print("DONE")


if __name__ == '__main__':
    main()
