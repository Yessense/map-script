import pickle
import string

# sys.path.append(".vsa")
from typing import List, Optional

import pandas as pd # type: ignore
from mapcore.swm.src.components.semnet import Sign, CausalMatrix, Event, Connector # type: ignore

from .random_script import Script, Step, Role, Bundle, SynSet, Roles, create_random_script # type: ignore
from ..preprocessing.wn import get_words_list
from .vsa import ItemMemory, HDVector # type: ignore
from ..vsa import vsa

DIMENSION = 1000


class ScriptEncoder:
    def __init__(self):
        self.alphabet_im = self.create_alphabet_im()
        self.scripts_im = ItemMemory(name='Script names',
                                     dimension=DIMENSION)
        self.roles_im = self.create_roles_im()
        self.roles_bundle_im = ItemMemory(name='Role Bundle', dimension=DIMENSION)
        self.full_roles_im = ItemMemory(name='Full Roles', dimension=DIMENSION)
        self.synsets_im = ItemMemory(name='Synsets', dimension=DIMENSION)
        self.words_im = ItemMemory(name='Words', dimension=DIMENSION)
        self.steps_im = ItemMemory(name='Steps', dimension=DIMENSION)
        self.synset_bundle_im = ItemMemory(name='Synset_bundle', dimension=DIMENSION)
        self.utils_im = ItemMemory(name='Utils', dimension=DIMENSION,
                                   init_vectors=['shift_step_v', 'shift_role_v', 'action_r',
                                                 'role_r', 'roles_r', 'end_v',
                                                 'synset_bundle_r'])

    def decode_script(self, name: str, decode_noise: bool = False) -> Script:
        # Support vectors
        script_v: HDVector = self.scripts_im[name]
        shift_v: HDVector = self.utils_im['shift_step_v']
        end_v: HDVector = self.utils_im['end_v']

        # Decode steps
        steps = []
        for i in range(100):
            step_v_noise = script_v / shift_v.cycle_shift(i + 2)
            if vsa.sim(step_v_noise, end_v) > 0.1:
                break

            step_name = self.steps_im.search(step_v_noise, names=True)
            step_v = self.steps_im[step_name]

            if decode_noise:
                step: Step = self.decode_step(step_v_noise, name=name, idx=i, decode_noise=decode_noise)
            else:
                step = self.decode_step(step_v, name=name, idx=i)
            steps.append(step)

        # Create script
        script: Script = Script(steps=steps, name=name)
        script.vector = script_v

        return script

    def decode_step(self, step_v: HDVector, name: str, idx: int, decode_noise: bool = False) -> Step:
        # role vectors
        action_r: HDVector = self.utils_im['action_r']
        roles_r: HDVector = self.utils_im['roles_r']

        # Action
        action_v_noise = step_v / action_r
        action_name = self.synsets_im.search(action_v_noise, names=True)
        action_v = self.synsets_im[action_name]
        sim_a = vsa.sim(action_v, action_v_noise)

        # Roles
        roles_v_noise = step_v / roles_r
        roles_name = self.roles_bundle_im.search(roles_v_noise, names=True)
        roles_v = self.roles_bundle_im[roles_name]
        sim_r = vsa.sim(roles_v, roles_v_noise)

        # Decode roles further
        action: SynSet = self.decode_synset(action_v, action_name)
        if decode_noise:
            roles: List[Role] = self.decode_roles(roles_v_noise, decode_noise=decode_noise)
        else:
            roles = self.decode_roles(roles_v, decode_noise=decode_noise)

        step: Step = Step(action=action, roles=roles, name=name, idx=idx)

        return step

    def decode_roles(self, roles_v: HDVector, decode_noise: bool = False) -> List[Role]:
        # find end vector
        end_v: HDVector = self.utils_im['end_v']
        shift_v: HDVector = self.utils_im['shift_role_v']

        roles: List[Role] = []
        for i in range(len(Roles)):
            roles_noise = roles_v / (shift_v.cycle_shift(i + 2))

            sim_end = vsa.sim(end_v, roles_noise)
            role_name = self.full_roles_im.search(roles_noise, names=True)
            role_v = self.full_roles_im[role_name]
            sim_r = vsa.sim(roles_noise, role_v)

            if sim_end >= sim_r:
                break

            if decode_noise:
                role: Role = self.decode_role(roles_noise)
            else:
                role  = self.decode_role(role_v)

            roles.append(role)
        return roles

    def decode_role(self, role_v: HDVector) -> Role:
        role_r = self.utils_im['role_r']
        synset_bundle_r = self.utils_im['synset_bundle_r']

        role_v_noise = role_v / role_r
        synset_bundle_v_noise = role_v / synset_bundle_r

        role_name = self.roles_im.search(role_v_noise, names=True)
        synset_bundle_name = self.synset_bundle_im.search(synset_bundle_v_noise, names=True)
        synset_bundle_v = self.synset_bundle_im[synset_bundle_name]

        role_enum: Roles = Roles(role_name)
        bundle: Bundle = self.decode_synset_bundle(synset_bundle_v)

        role: Role = Role(role=role_enum, bundle=bundle)

        return role

    def decode_synset_bundle(self, synset_bundle_v) -> Bundle:
        words = self.synsets_im.search(synset_bundle_v, n_top=None, threshold=0.2, names=True)
        synsets: List[SynSet] = []
        for word in words:
            synset_v = self.synsets_im[word]
            synsets.append(self.decode_synset(synset_v, word))

        bundle = Bundle(synsets)
        return bundle

    def decode_synset(self, synset_v, word) -> SynSet:
        words = self.words_im.search(synset_v, n_top=None, threshold=0.1, names=True)
        synset = SynSet(word=word,
                        words_list=words)
        synset.vector = self.synsets_im[word]
        return synset

    def encode_script_sign(self, script: Sign) -> None:
        significance: CausalMatrix
        significance_index: int
        for significance_index, significance in script.significances.items():
            self.encode_script_cm(significance_index, significance)

    def encode_script_cm(self, script_index: int, cm: CausalMatrix) -> HDVector:
        """ Script = [sum(shift * script_step)]"""
        steps: List[HDVector] = []
        shift_v: HDVector = self.utils_im['shift_step_v']
        end_v: HDVector = self.utils_im['end_v']

        script_name: str = f'{cm.sign.name}:{script_index}'

        step: Event
        for i, step in enumerate(cm.cause):
            step_v = self.encode_script_step(step=step, script_name=script_name, step_index=i)
            steps.append(step_v * shift_v.cycle_shift(2 + i))

        # mark end of sequence by adding end vector
        steps.append(end_v * shift_v.cycle_shift(2 + len(cm.cause)))

        script_v = vsa.bundle(*steps)
        self.scripts_im[script_name] = script_v
        return script_v

    def encode_script_step(self, step: Event, script_name: str, step_index: int) -> HDVector:
        """ Script step = [action_role * action_v + roles_role * roles_v]"""

        step_name: str = f'{script_name}:{step_index}'
        # store result
        roles: List[HDVector] = []

        # util vectors
        action_r: HDVector = self.utils_im['action_r']
        roles_r: HDVector = self.utils_im['roles_r']
        end_v: HDVector = self.utils_im['end_v']
        shift_role_v: HDVector = self.utils_im['shift_role_v']

        # encode action
        action_name: str
        action_significance_number: int

        # only one coincident - script step
        connector: Connector = list(step.coincidences)[0]
        action_sign: Sign = connector.out_sign
        action_name = action_sign.name

        significance_index: int
        significance: CausalMatrix
        for significance_index, significance in enumerate(action_sign.significances.values()):
            # only one significance CM is used for representing action significance
            used_significance = False

            role_event: Event
            for role_index, role_event in enumerate(significance.cause):
                if not len(role_event.coincidences):
                    continue
                else:
                    used_significance = True

                    # bundle roles multiplied by action vector.
                    # action vector permuted by 2 + i times because shift is ** function
                    role_v = self.encode_role(role_index, role_event, step_name)
                    roles.append(role_v * shift_role_v.cycle_shift(2 + role_index))

            if used_significance:
                action_significance_number = significance_index
                break

        action_v = self.encode_synset_v(action_name, action_significance_number)
        # mark end of sequence by adding end vector
        roles.append(end_v * shift_role_v.cycle_shift(2 + len(roles)))
        roles_v = vsa.bundle(*roles)
        self.roles_bundle_im[step_name] = roles_v

        step_v = vsa.bundle(action_r * action_v, roles_r * roles_v)
        step.vector = step_v
        self.steps_im[step_name] = step_v
        return step_v

    def encode_role(self, role_index: int, role_event: Event, step_name: str) -> HDVector:
        """ Role = [role_role * role_v + synset_bundle_role * synset_bundle_v]"""

        role_name: str = list(Roles)[role_index].value

        role_r: HDVector = self.utils_im['role_r']
        synset_bundle_r: HDVector = self.utils_im['synset_bundle_r']
        role_v: HDVector = self.roles_im[role_name]

        vectors: List[HDVector] = []
        names: List[str] = []
        connector: Connector
        for connector in role_event.coincidences:
            obj_name: str = connector.out_sign.name
            obj_number: int = connector.out_index

            names.append(f'{obj_name}:{obj_number}')

            vector = self.encode_synset_v(obj_name, obj_number)
            vectors.append(vector)

        synset_bundle_v: HDVector = vsa.bundle(*vectors)
        synset_bundle_name = " ".join(names)
        if self.synset_bundle_im[synset_bundle_name] is None:
            self.synset_bundle_im[synset_bundle_name] = synset_bundle_v

        full_role_v: HDVector = vsa.bundle(role_r * role_v, synset_bundle_r * synset_bundle_v)
        roles_name: str = f'{step_name}:{role_name}'
        self.full_roles_im[roles_name] = full_role_v

        return full_role_v

    def encode_synset_v(self, name: str, number: int) -> HDVector:
        """ Create HD vector of synset and add it to memories"""
        synset_name = f'{name}:{number}'
        synset_v: Optional[HDVector] = self.synsets_im[synset_name]

        # create sysnset vector if not exist
        if synset_v is None:
            words_list = get_words_list(name, number)
            vectors = []

            # synset = [sum(word)]
            for word in words_list:
                vector = self.words_im[word]

                # create words from alphabet if not exist
                if vector is None:
                    vector = self.word_to_vector(word, self.alphabet_im)
                    self.words_im[word] = vector

                vectors.append(vector)
            synset_v = vsa.bundle(*vectors)
            self.synsets_im[synset_name] = synset_v

        return synset_v

    @staticmethod
    def create_roles_im() -> vsa.ItemMemory:
        roles_im = vsa.ItemMemory(name='Roles', dimension=DIMENSION)
        roles_im.append_batch([role.value for role in Roles])
        return roles_im

    @staticmethod
    def create_alphabet_im() -> ItemMemory:
        """Create Alphabet item memory"""
        alphabet_im = ItemMemory(name='Alphabet', dimension=DIMENSION)
        alphabet_im.append_batch(names=list(string.ascii_letters + string.punctuation + string.digits + '’'))
        return alphabet_im

    @staticmethod
    def word_to_vector(word: str, alphabet_im: ItemMemory) -> HDVector:
        """Encode word by binding letters together"""
        vector: HDVector = alphabet_im[word[0]]

        shift = 0
        for c in word[1:]:
            shift += 1
            vector = vector * alphabet_im[c].cycle_shift(shift)
        return vector