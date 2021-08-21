from typing import List, Tuple, Dict, Any, Union

from mapcore.swm.src.components.semnet import Sign, Event

from src.script_extraction.text_preprocessing.combine_actions_with_clusters import combine_actions_with_clusters
from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters, resolve_pronouns
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions
from src.script_extraction.text_preprocessing.words_object import Roles, Action, Cluster, Obj, WordsObject
from src.text_info_restaurant import create_text_info_restaurant


class Script:
    def __init__(self, text_info: Dict[str, Any]):
        self.text_info: Dict[str, Any] = text_info
        self.sign: Sign = Sign("Script")
        self.actions: List[Sign] = []
        self.actions_significance_number: List[int] = []
        self.objects: Dict[str, Sign] = dict()
        self.role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}

        self.create_signs()

    def _add_action_sign(self, action: Action) -> None:
        name = action.lemma

        sign = Sign(name)

        # add meanings and significances for each wn meaning
        for i in range(action.synsets_len):
            # add signifincances
            significance = sign.add_significance(pm=None)

            # Creating place for adding images
            image = sign.add_image(pm=None)
            image.add_event(event=Event(order=0))

            # Events contains roles
            for i in range(len(self.role_int)):
                significance.add_event(event=Event(order=i))

        self.actions.append(sign)
        self.actions_significance_number.append(action.synset_number)

    def _add_object_sign(self, obj: Union[WordsObject, Obj]) -> None:
        if obj.lemma in self.objects:
            return
        name = obj.lemma

        sign = Sign(name)

        # add meanings and significances for each wn meaning
        for i in range(obj.synsets_len):
            # add signifincances
            significance = sign.add_significance(pm=None)

            # Creating place for adding images
            image = sign.add_image(pm=None)
            image.add_event(event=Event(order=0))

        self.objects[name] = sign

    def _process_object(self, obj: Union[WordsObject, Obj]) -> None:
        if obj.cluster is None:
            self._add_object_sign(obj)
            for image in obj.images:
                self._add_object_sign(image)
        else:
            for cluster_obj in obj.cluster.objects:
                self._add_object_sign(cluster_obj)
                for image in cluster_obj.images:
                    self._add_object_sign(image)

    def create_signs(self):
        # Information preparation
        actions: List[Action] = extract_actions(self.text_info)
        clusters: List[Cluster] = extract_clusters(self.text_info)
        combine_actions_with_clusters(actions, clusters, self.text_info)
        resolve_pronouns(clusters)

        # Add signs to script
        for action in actions:
            # if no roles, look next action
            if not action.is_script_step() or not action.has_valid_meanings:
                continue
            self._add_action_sign(action=action)

            for obj in action.objects:
                self._process_object(obj)
        print("Done")


def main():
    text_info = create_text_info_restaurant()

    script = Script(text_info)
    script.create_signs()
    print("Done")


if __name__ == '__main__':
    main()
