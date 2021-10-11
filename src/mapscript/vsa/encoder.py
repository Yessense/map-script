from typing import List, Optional, Iterable

import numpy as np
from mapcore.swm.src.components.semnet import Sign

from ..script import Script
from . import vsa
from ..preprocessing.words_object import Roles


class ScriptEncoder:
    def __init__(self, script: Script, random_state: Optional[int] = None):
        # for repeatable results
        if random_state is not None:
            self.random_state: np.random.RandomState = np.random.RandomState(seed=random_state)
        self.script: Script = script

        self.item_memory: vsa.ItemMemory = vsa.ItemMemory(name='Scripts')
        # max_shift for searching in item memory

        self.role_v: np.ndarray = vsa.generate(self.random_state)
        self.script_v: np.ndarray = vsa.generate(self.random_state)

        self.item_memory.append(f'_ROLE_V', self.role_v)
        self.item_memory.append(f'_SCRIPT_V', self.script_v)

        self._encode_signs(self.script.actions_signs)
        self._encode_signs(self.script.objects_signs.values())

    def _encode_signs(self, signs: Iterable[Sign]):
        """
        Add vector + bundle of significances to item memory
        """
        sign: Sign
        for sign in signs:
            if sign.name not in self.item_memory.get_names():
                sign_v = vsa.generate(self.random_state)
                significances_bundle: List[np.ndarray] = []

                for i in range(len(sign.significances)):
                    significances_bundle.append(vsa.cycle_shift(sign_v, i + 1))

                sign_v_bundle = vsa.bundle2(significances_bundle)

                self.item_memory.append(sign.name, sign_v)
                self.item_memory.append(f'{sign.name}_significances', sign_v_bundle)

    def check_vectors(self, sign_name, significance_n):
        vector = self.item_memory.get_vector(sign_name)
        shift_vect = vsa.cycle_shift(vector, significance_n)

        sim = self.item_memory.search(shift_vect, distance=False)
        name = self.item_memory.get_name(np.argmax(sim))

        return name

