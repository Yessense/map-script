import dataclasses
import itertools
from itertools import combinations
from typing import List, Tuple, Dict, Any, Union, Optional, Set

import networkx as nx
from mapcore.swm.src.components.semnet import Sign, Event, CausalMatrix, Connector
from nltk.corpus.reader import Synset

from .preprocessing.wn import get_synsets, get_hypernyms, get_synset_number
from .preprocessing.combine_actions_with_clusters import combine_actions_with_clusters
from .preprocessing.extract_clusters import extract_clusters, resolve_pronouns
from .preprocessing.extract_semantic_roles import extract_actions
from .preprocessing.words_object import Roles, Action, Cluster, Obj, WordsObject
from .samples.text_info.text_info_restaurant import create_text_info_restaurant


@dataclasses.dataclass
class SynObj:
    lemma: str
    number: int
    ss: Dict[str, Synset] = dataclasses.field(default_factory=dict)
    hypernyms: Dict[str, Dict[str, Synset]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.set_synset()
        self.add_hypernyms()

    def set_synset(self):
        self.ss = get_synsets(self.lemma)

    def add_hypernyms(self):
        for name, ss in self.ss.items():
            self.hypernyms[name] = get_hypernyms(ss)


class Script:
    def __init__(self, text_info: Dict[str, Any]):
        self._text_info: Dict[str, Any] = text_info
        self._role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}
        self.possible_roles = [0, 1, 2]

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
        self.resolve_hypernyms_hyponyms()

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
                    if role_object.has_valid_meanings:
                        self._add_role_object_to_action(action_sign=action_sign,
                                                        action=action,
                                                        role_sign=self.objects_signs[role_object.lemma],
                                                        role_object=role_object)
                # Role has cluster of fillers
                else:
                    for role_cluster_object in role_object.cluster.objects:
                        if role_cluster_object.lemma not in self.objects_signs:
                            self._add_object_sign(role_cluster_object)
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

    def resolve_hypernyms_hyponyms(self):
        for action in self.actions_signs:
            significance: CausalMatrix
            for significance in action.significances.values():
                event: Event
                # checking each role in action
                for event in significance.cause:
                    # if 2 or more
                    if len(event.coincidences) >= 2:
                        # get all fillers -> Dict[str, SynObj]
                        fillers: Dict[str, SynObj] = dict()

                        connector: Connector
                        for connector in event.coincidences:
                            syn_obj = SynObj(lemma=connector.out_sign.name,
                                             number=connector.out_index - 1)
                            fillers[f'{syn_obj.lemma}:{syn_obj.number}'] = syn_obj

                        # get list of changes
                        l1, l2 = self.get_neighbours(fillers)
                        self.process_l1(l1, event, significance)

    def create_synset_signs(self, synset: Synset,
                            event: Event,
                            significance: CausalMatrix):
        for lemma in synset.lemma_names():
            ss_number, ss_len = get_synset_number(lemma, synset.name())
            if lemma in self.objects_signs:
                sign = self.objects_signs[lemma]
            else:
                sign = Sign(lemma)

                # add meanings and significances for each wn meaning
                for i in range(ss_len):
                    # add signifincances
                    sign.add_significance(pm=None)

                    # Creating place for adding images
                    image = sign.add_image(pm=None)
                    image.add_event(event=Event(order=0))

                self.objects_signs[lemma] = sign
            connector: Connector = Connector(in_sign=significance.sign,
                                             out_sign=sign,
                                             in_index=significance.index,  # action signifincance number
                                             out_index=ss_number + 1,
                                             # role significance number
                                             in_order=event.order)  # role number (Event number)
            event.add_coincident(base='significance', connector=connector)

    def process_l1(self, l1: List[Tuple[Synset, Synset]],
                   event: Event,
                   significance: CausalMatrix):
        for synset, hypernym in l1:
            self.create_synset_signs(synset, event, significance)
            self.create_synset_signs(hypernym, event, significance)




    def get_neighbours(self, fillers: Dict[str, SynObj]):
        # one value is hypernym for another
        l1: List[Tuple[Synset, Synset]] = []
        # two values has the same hypernym
        l2: List[Tuple[Synset, Synset, Synset]] = []
        # coreference values f.e [man, person]
        for name, syn_obj in fillers.items():
            # for semantic value f.e. man.n.01
            for ss_name, synset in syn_obj.ss.items():
                hypernyms: Dict[str, Synset] = syn_obj.hypernyms[ss_name]
                for h_name, hypernym in hypernyms.items():
                    for name_1, syn_obj_1 in fillers.items():
                        if name_1 != name:
                            # if hypernym contains man.n.01
                            if h_name in syn_obj_1.ss:
                                l1.append((synset, hypernym))
                            for ss_name_1, synset_1 in syn_obj_1.ss.items():
                                if h_name in syn_obj_1.hypernyms[ss_name_1]:
                                    l2.append((synset, synset_1, hypernym))
                                    # TODO: remove double check

        return l1, l2


def main():
    text_info = create_text_info_restaurant()

    script = Script(text_info)
    print("Done")


if __name__ == '__main__':
    main()
