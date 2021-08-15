import enum
from typing import Any, Dict, List, Union, Iterable

from mapcore.swm.src.components.semnet import Sign, Event, CausalMatrix, Connector

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, \
    combine_actions_with_clusters
from src.script_extraction.text_preprocessing.words_object import Action, Cluster, WordsObject, Obj, Roles
from src.text_info_restaurant import create_text_info_restaurant




def add_new_sign(script: Dict[str, Sign],
                 obj: Union[WordsObject, Obj, Action],
                 roles_signs: List[Sign]) -> Dict[str, Sign]:
    name = obj.lemma

    if name not in script:
        sign = Sign(name)

        for i in range(obj.synsets_len):
            significance = sign.add_significance(pm=None)

            if isinstance(obj, Action):
                for i, role in enumerate(roles_signs):
                    significance.add_event(event=Event(order=i))


        script[name] = sign

    else:
        if isinstance(obj, Action):
            sign = script[name]
            if len(sign.significances[1].cause) != obj.synsets_len:
                for significance_number in sign.significances:
                    for i, role in enumerate(roles_signs):
                        sign.significances[significance_number].add_event(event=Event(order=i))

    return script


def create_roles_signs(roles: enum) -> List[Sign]:
    roles_signs: List[Sign] = []
    for role in roles:
        roles_signs.append(Sign(role.value))
    return roles_signs


def extract_script(text_info: Dict[str, Any]):
    script: Dict[str, Sign] = dict()

    # Information preparation
    actions: List[Action] = extract_actions(text_info)
    clusters: List[Cluster] = extract_clusters(text_info)
    combine_actions_with_clusters(actions, clusters, text_info)

    # All possible roles
    roles_signs = create_roles_signs(Roles)
    roles_dict = {role: i for i, role in enumerate(Roles)}

    # Add signs to script
    for action in actions:
        add_new_sign(script=script, obj=action, roles_signs=roles_signs)
        for obj in action.objects:
            add_new_sign(script=script, obj=obj, roles_signs=roles_signs)
            for image in obj.images:
                add_new_sign(script=script, obj=image, roles_signs=roles_signs)


    for action in actions:
        action_sign = script[action.lemma]

        for role_object in action.objects:
            if role_object.synsets_len != -1:
                role_sign = script[role_object.lemma]
                action_cm: CausalMatrix = action_sign.significances[action.synset_number + 1]
                # action_cm.add_feature(script[role_object.lemma].significances[role_object.synset_number + 1])
                action_event: Event = action_cm.cause[roles_dict[role_object.arg_type]]
                connector: Connector = Connector(in_sign=action_sign,
                                                 out_sign=role_sign,
                                                 in_index=0)
                action_event.add_coincident(base='significance', connector=connector)

    for action in actions:
        for obj in action.objects:
            for image in obj.images:
                pass

    return script


def main():
    text_info = create_text_info_restaurant()

    script = extract_script(text_info)

    print("DONE")


if __name__ == '__main__':
    main()
