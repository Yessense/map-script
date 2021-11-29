from typing import List, Optional, Iterable

import numpy as np
from mapcore.swm.src.components.semnet import Sign, CausalMatrix, Event, Connector

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
        self.step_v: np.ndarray = vsa.generate(self.random_state)

        self.item_memory.append(f'_ROLE_V', self.role_v)
        self.item_memory.append(f'_SCRIPT_V', self.script_v)
        self.item_memory.append(f'_STEP_V', self.script_v)

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

    def encode_script(self):
        script_vectors = []

        # scripts
        significance: CausalMatrix
        for significance_index, significance in self.script.sign.significances.items():
            script_v = vsa.cycle_shift(self.script_v, significance_index)
            steps: List[np.ndarray] = []

            # steps
            event: Event
            for event_index, event in enumerate(significance.cause):
                step_v = vsa.cycle_shift(self.step_v, event_index)

                # only one coincident - script step
                connector: Connector = list(event.coincidences)[0]
                action_sign: Sign = connector.out_sign
                name: str = action_sign.name
                shift: int = connector.out_index

                action_v = self.item_memory.get_vector(name)
                action_v = vsa.cycle_shift(action_v, shift)

                roles_bundle = []
                # roles
                a_significance: CausalMatrix
                for a_significance_index, a_significance in self.script.sign.significances.items():

                    one_signif_for_action = False
                    role_event: Event
                    for role_index, role_event in enumerate(a_significance.cause):
                        if not len(role_event.coincidences):
                            continue
                        else:
                            one_signif_for_action = True
                            role_bundle: List[np.ndarray] = []

                            # bundle all fillers
                            conn: Connector
                            for conn in role_event.coincidences:
                                obj_sign: Sign = connector.out_sign
                                name: str = obj_sign.name
                                shift: int = connector.out_index

                                obj_v = self.item_memory.get_vector(name)
                                obj_v = vsa.cycle_shift(obj_v, shift)

                                role_bundle.append(obj_v)

                            # role fillers bundle hd vector
                            role_bundle_v: np.ndarray = vsa.bundle2(role_bundle)
                            # role index hd vector
                            role_v: np.ndarray = vsa.cycle_shift(self.role_v, role_index)
                            # role * role fillers
                            role_pool_v: np.ndarray = vsa.bind(role_bundle_v, role_v)

                            # collecting a set of role pools
                            roles_bundle.append(role_pool_v)
                    if one_signif_for_action:
                        break
                roles_bundle_v = vsa.bundle2(roles_bundle)

                action_roles_v = vsa.bind(action_v, roles_bundle_v)
                step_action_roles_v = vsa.bind(step_v, action_roles_v)
                steps.append(step_action_roles_v)

            steps_bundle_v = vsa.bundle2(steps)
            script_steps_v = vsa.bind(script_v, steps_bundle_v)
            script_vectors.append(script_steps_v)





















