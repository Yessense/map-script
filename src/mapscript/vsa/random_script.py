import dataclasses
from enum import Enum
from typing import Tuple, List, Set, Optional
import random
from itertools import count

import numpy as np

from ..vsa import vsa
from .vsa import HDVector
from nltk.corpus import wordnet as wn


class Roles(Enum):
    """
    Semantic roles enum
    """
    ARG0 = 'ARG0'
    ARG1 = 'ARG1'
    ARG2 = 'ARG2'
    ARG3 = 'ARG3'
    ARG4 = 'ARG4'
    ARGM_TMP = 'ARGM-TMP'
    ARGM_DIR = 'ARGM-DIR'
    ARGM_DIS = 'ARGM-DIS'
    ARGM_EXT = 'ARGM-EXT'
    ARGM_LOC = 'ARGM-LOC'
    ARGM_MNR = 'ARGM-MNR'
    ARGM_MOD = 'ARGM-MOD'
    ARGM_NEG = 'ARGM-NEG'
    ARGM_PRD = 'ARGM-PRD'
    ARGM_PRP = 'ARGM-PRP'
    ARGM_ADV = 'ARGM-ADV'
    ARGM_GOL = 'ARGM-GOL'
    ARGM_CAU = 'ARGM-CAU'
    ARGM_REC = 'ARGM-REC'
    ARGM_COM = 'ARGM-COM'
    R_ARGM_TMP = 'R-ARGM-TMP'
    R_ARGM_LOC = 'R-ARGM-LOC'
    R_ARG0 = 'R-ARG0'
    R_ARG1 = 'R-ARG1'
    R_ARG2 = 'R-ARG2'
    C_ARG1 = 'C-ARG1'
    C_ARG2 = 'C-ARG2'
    V = 'V'
    NONE = "NONE"
    NAMED_GROUP = "NAMED-GROUP"


ACTIONS: List[str] = ['have', 'book', 'find', 'see', 'bottle', 'cook', 'spoil', 'show', 'get', 'include', 'complain',
                      'consume', 'look', 'think', 'start', 'come', 'serve', 'choose', 'order', 'manage', 'decide',
                      'judge', 'bake', 'advise', 'be', 'mean', 'make', 'overlook', 'hear', 'price', 'worry', 'run',
                      'open']
OBJECTS: List[str] = ['delight', 'way', 'include', 'mean', 'vegetable', 'chef', 'french', 'week', 'little', 'freezer',
                      'time', 'kitchen', 'favourable', 'list', 'bake', 'good', 'prepare', 'wardle', 'comment',
                      'premise', 'speciality', 'wine', 'fresh', 'bread', 'bill', 'coffee', 'taste', 'over', 'underdone',
                      'extensive', 'steak', 'class', 'quality', 'reasonably', 'people', 'choice', 'rather',
                      'acceptable', 'wife', 'often', 'salad', 'enough', 'menu', 'lamb', 'best', 'early', 'main',
                      'excellent', 'table', 'trout', 'cook', 'restaurant', 'satisfactory', 'get', 'nice', 'always',
                      'long', 'be', 'service', 'traditional', 'one', 'perfect', 'look', 'advise', 'difficulty',
                      'business', 'number', 'usual', 'however', 'welcome', 'family', 'portion', 'pound', 'column',
                      'newly', 'large', 'helping', 'mill', 'course', 'dean', 'small', 'second']


class Vector:
    def __init__(self):
        self._vector: Optional[HDVector] = None

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, hd_vector: HDVector):
        if self._vector is None:
            self._vector = hd_vector


class SynSet(Vector):
    def __init__(self, word: str, words_list: List[str]):
        super().__init__()
        self.word = word
        self.words_list = words_list

    @property
    def name(self):
        if self.words_list is None:
            return ""
        else:
            return " ".join(self.words_list)

    def __repr__(self):
        return repr(self.name)

    def sim(self, other) -> float:
        assert len(self.words_list) and len(other.words_list)
        intersection = len(set(self.words_list).intersection(other.words_list))
        union = len(set(self.words_list).union(other.words_list))
        iou = intersection / union
        assert iou <= 1
        return iou

    def __eq__(self, other):
        if set(self.words_list) == set(other.words_list):
            return True
        return False


class Bundle(Vector):
    def __init__(self, items: List[SynSet]):
        super().__init__()
        self.items = items

    def __repr__(self):
        return repr(self.items)

    def sim(self, other, check_synset=False) -> float:
        assert len(self.items) and len(other.items)

        intersection: float = 0
        count: float = 0

        # for creating decoded bundle we already using parametrized matching
        # so now we check for a full match
        for i, item_r in enumerate(self.items):
            for j, item_c in enumerate(other.items):
                if vsa.sim(item_r.vector, item_c.vector) > 0.99:
                    intersection += item_r.sim(item_c)
                    count += 1.
                    break

        assert intersection <= other.size
        union = self.size + other.size - count
        if check_synset:
            iou = intersection / union
        else:
            iou = count / union
        assert iou <= 1
        return iou

    @property
    def size(self) -> int:
        return len(self.items)


class Role(Vector):
    def __init__(self, role: Roles, bundle: Bundle):
        super().__init__()
        self.role: Roles = role
        self.bundle: Bundle = bundle

    def __repr__(self):
        return repr(self.role.value)

    def sim(self, other, check_synset=False) -> float:
        if self.role == other.role:
            similarity = self.bundle.sim(other.bundle, check_synset=check_synset)
            assert similarity <= 1
            return similarity
        else:
            return 0.


class Step(Vector):
    def __init__(self, action: SynSet, roles: List[Role], name: str, idx: int):
        super().__init__()
        self.action = action
        self.roles: List[Role] = roles
        self._script_name = name
        self.idx = idx

    @property
    def name(self):
        return f'{self._script_name}:{self.idx}'

    def __repr__(self):
        return f'{self.action!r} {self.roles!r}'

    def sim(self, noise_step,
            check_role=False,
            check_synset=False) -> float:
        """ Compare valid script to noise"""

        # check for empty step for zero division
        if not len(self.roles) or not len(noise_step.roles):
            return 0.

        action_sim = vsa.sim(self.action.vector, noise_step.action.vector)
        if action_sim < 0.99:
            return 0

        # if action_sim < 1:
        #     print(action_sim)
        #     print(f'Source action: {self.action.words_list!r}')
        #     print(f'Target action: {noise_step.action.words_list!r}')

        # actions have similarity
        intersection: float = 0.
        count: float = 0.

        for role_r in self.roles:
            max_sim = 0
            for role_c in noise_step.roles:
                role_sim = role_r.sim(role_c, check_synset=check_synset)
                max_sim = max(max_sim, role_sim)
            # similar role found
            if max_sim > 0:
                count += 1.
                intersection += max_sim

        union = len(self.roles) + len(noise_step.roles) - count

        if check_role:
            iou = intersection / union
        else:
            iou = count / union

        assert iou <= 1
        return iou


class Script(Vector):
    _ids = count(0)

    def __init__(self, steps: List[Step], name):
        super().__init__()
        self.id = next(self._ids)
        self.steps: List[Step] = steps
        self._name = name

    @property
    def name(self):
        return f'{self._name}:{self.id}'

    def sim(self, noise_script, check_role=False, check_synset=False) -> float:
        assert len(self.steps) == len(noise_script.steps)

        similarity: float = 0.
        for step_r, step_c in zip(self.steps, noise_script.steps):
            similarity += step_r.sim(step_c, check_role=check_role, check_synset=check_synset)

        similarity /= len(self.steps)
        return similarity


def create_random_script(n_steps: Tuple[int, int], n_roles: Tuple[int, int], n_bundle_items: Tuple[int, int]) -> Script:
    steps: List[Step] = []
    name = 'RandomScript'

    script = Script(steps=steps, name=name)

    for i in range(random.randint(*n_steps)):
        step = create_random_step(n_roles, n_bundle_items, name=script.name, idx=i)
        script.steps.append(step)

    return script


def create_random_step(n_roles: Tuple[int, int], n_bundle_items: Tuple[int, int], name: str, idx: int) -> Step:
    action = create_random_action()
    step_roles: List[Role] = []

    k = random.randint(*n_roles)
    roles = random.sample(list(Roles), k=k)

    for r in roles:
        role = create_random_role(r, n_bundle_items)
        step_roles.append(role)

    step = Step(action=action,
                roles=step_roles,
                name=name,
                idx=idx)
    return step


def create_random_role(role: Roles, n_bundle_items) -> Role:
    bundle: Bundle = create_random_bundle(n_bundle_items)
    r = Role(role=role, bundle=bundle)
    return r


def create_random_bundle(n_bundle_items) -> Bundle:
    items: List[SynSet] = []
    sample = random.sample(OBJECTS, random.randint(*n_bundle_items))
    for word in sample:
        obj: SynSet = create_random_object(word)

        # if somehow created 2 similar synsets, we omit one of them
        if is_obj_already_exist(items, obj):
            continue
        items.append(obj)
    bundle = Bundle(items)
    return bundle


def is_obj_already_exist(items: List[SynSet], obj: SynSet):
    for item in items:
        if item == obj:
            return True
    return False


def create_random_synset(word: str, samples: List[str]) -> SynSet:
    synsets = wn.synsets(word)

    length = len(synsets)
    if not length:
        words_list = [word]
    else:
        words_list = synsets[random.randint(0, length - 1)].lemma_names()
    synset = SynSet(word=word, words_list=words_list)
    return synset


def create_random_action() -> SynSet:
    action = create_random_synset(word=random.choice(ACTIONS), samples=ACTIONS)
    return action


def create_random_object(word: str) -> SynSet:
    obj = create_random_synset(word, OBJECTS)
    return obj


if __name__ == '__main__':
    random_script = create_random_script(n_steps=(20, 20),
                                         n_roles=(5, 5),
                                         n_bundle_items=(15, 15))
    # print("Done")
