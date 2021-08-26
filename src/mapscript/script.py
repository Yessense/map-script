import itertools
from itertools import combinations
from typing import List, Tuple, Dict, Any, Union, Optional, Set

import networkx as nx
from mapcore.swm.src.components.semnet import Sign, Event, CausalMatrix, Connector

from .preprocessing.combine_actions_with_clusters import combine_actions_with_clusters
from .preprocessing.extract_clusters import extract_clusters, resolve_pronouns
from .preprocessing.extract_semantic_roles import extract_actions
from .preprocessing.words_object import Roles, Action, Cluster, Obj, WordsObject
from .samples.text_info.text_info_restaurant import create_text_info_restaurant


class Script:
    def __init__(self, text_info: Dict[str, Any]):
        self._text_info: Dict[str, Any] = text_info
        self._role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}
        self.possible_roles = [0, 1]

        self._actions: List[Action] = []
        self._clusters: List[Cluster] = []
        self._actions_significance_number: List[int] = []
        self._connected_pairs: Set[Tuple[int, ...]] = set()

        self.actions_signs: List[Sign] = []
        self.objects_signs: Dict[str, Sign] = dict()

        self.sign: Sign = Sign("Script")

        self.create_signs()
        self.add_roles()
        self.add_images()
        self.create_script()

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
            for i in range(len(self._role_int)):
                significance.add_event(event=Event(order=i))

        self.actions_signs.append(sign)
        self._actions_significance_number.append(action.synset_number)

    def _add_object_sign(self, obj: Union[WordsObject, Obj]) -> None:
        if obj.lemma in self.objects_signs or not obj.has_valid_meanings:
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

        self.objects_signs[name] = sign

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
        self._actions = extract_actions(self._text_info)
        self._clusters: List[Cluster] = extract_clusters(self._text_info)
        combine_actions_with_clusters(self._actions, self._clusters, self._text_info)
        resolve_pronouns(self._clusters)

        # Add signs to script
        for action in self._actions:
            # if no roles, look next action
            if not action.is_script_step() or not action.has_valid_meanings:
                continue
            self._add_action_sign(action=action)

            for obj in action.objects:
                self._process_object(obj)

    def _add_role_object_to_action(self,
                                   action_sign: Sign,
                                   action: Action,
                                   role_sign: Sign,
                                   role_object: Obj,
                                   arg_type: Optional[Roles] = None) -> None:
        if not role_object.has_valid_meanings:
            return
        if arg_type is None:
            arg_type = role_object.arg_type

        role_number = self._role_int[arg_type]

        action_cm: CausalMatrix = action_sign.significances[action.synset_number + 1]
        action_event: Event = action_cm.cause[role_number]
        connector: Connector = Connector(in_sign=action_sign,
                                         out_sign=role_sign,
                                         in_index=action.synset_number + 1,  # action signifincance number
                                         out_index=role_object.synset_number + 1,
                                         # role significance number
                                         in_order=role_number)  # role number (Event number)
        action_event.add_coincident(base='significance', connector=connector)

    def add_roles(self):
        """
        Add role fillers to _actions
        """
        i: int = 0
        for action in self._actions:
            if not action.is_script_step() or not action.has_valid_meanings:
                continue
            action_sign = self.actions_signs[i]
            i += 1

            for role_object in action.objects:
                # Only one role
                if role_object.cluster is None:
                    self._add_role_object_to_action(action_sign=action_sign,
                                                    action=action,
                                                    role_sign=self.objects_signs[role_object.lemma],
                                                    role_object=role_object)
                # Role has cluster of fillers
                else:
                    for role_cluster_object in role_object.cluster.objects:
                        self._add_role_object_to_action(action_sign=action_sign,
                                                        action=action,
                                                        role_sign=self.objects_signs[role_cluster_object.lemma],
                                                        role_object=role_cluster_object,
                                                        arg_type=role_object.arg_type)

    def add_images(self):
        for action in self._actions:
            if not action.is_script_step() or not action.has_valid_meanings:
                continue
            for obj in action.objects:
                if obj.cluster is None:
                    for image in obj.images:
                        self._add_image_to_object(obj, image)
                else:
                    for image in obj.cluster.images.values():
                        self._add_image_to_object(obj, image)

    def _add_image_to_object(self, obj: Obj, image: WordsObject) -> None:
        if not image.has_valid_meanings:
            return

        obj_sign = self.objects_signs[obj.lemma]
        image_sign = self.objects_signs[image.lemma]

        obj_image_matrix: CausalMatrix = obj_sign.images[obj.synset_number + 1]
        obj_image_event: Event = obj_image_matrix.cause[0]

        # connector to image
        connector: Connector = Connector(in_sign=obj_sign,
                                         out_sign=image_sign,
                                         in_index=obj.synset_number + 1,  # action images number
                                         out_index=image.synset_number + 1,  # image images number
                                         in_order=0)
        obj_image_event.add_coincident(base='image', connector=connector)

    def find_connected_pairs(self) -> Set[Tuple[int, ...]]:
        pairs: Set[Tuple[int, ...]] = set()

        for object_sign in self.objects_signs.values():
            in_signs: List[Sign] = []
            # look for all actions, connected to this object
            for connector in object_sign.out_significances:
                if connector.in_order not in self.possible_roles:
                    continue
                already_in: bool = False
                for sign in in_signs:
                    if connector.in_sign is sign:
                        already_in = True
                    break
                if not already_in:
                    in_signs.append(connector.in_sign)
            # get actions numbers
            in_numbers = []
            for in_sign in in_signs:
                for i, act_sign in enumerate(self.actions_signs):
                    if act_sign is in_sign:
                        in_numbers.append(i)
            # create pairs
            for elem in combinations(in_numbers, 2):
                pairs.add(elem)
        self._connected_pairs = pairs
        return pairs

    def create_script(self, limit: Optional[int] = None) -> Sign:
        pairs = self.find_connected_pairs()

        G: nx.Graph = nx.Graph(list(pairs))

        # connected components in graph
        components: List = sorted(list(nx.connected_components(G)), key=len, reverse=True)

        # add signle nodes
        single_nodes = [{node_number} for node_number
                        in range(len(self.actions_signs))
                        if node_number not in list(itertools.chain(*components))]
        components += single_nodes

        # restriction to script nodes length
        if limit is None:
            limit = len(components)
        for component in components[:limit]:
            cm = self.sign.add_significance()
            for sign_index in component:
                action_sign = self.actions_signs[sign_index]
                cm_index: int = self._actions_significance_number[sign_index]
                connector = cm.add_feature(action_sign.significances[cm_index + 1])
                action_sign.add_out_significance(connector)
        return self.sign


def main():
    text_info = create_text_info_restaurant()

    script = Script(text_info)
    print("Done")


if __name__ == '__main__':
    main()
