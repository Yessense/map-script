from typing import List, Tuple, Dict, Any

from mapcore.swm.src.components.semnet import Sign

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters, resolve_pronouns
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, \
    combine_actions_with_clusters
from src.script_extraction.text_preprocessing.words_object import Roles, Action, Cluster


class Script:
    def __init__(self, text_info: Dict[str, Any]):
        self.text_info: Dict[str, Any] = text_info
        self.sign: Sign = Sign("Script")
        self.actions: List[Tuple[Sign, int]] = []
        self.objects: Dict[str, Sign] = dict()
        self.role_int: Dict[Roles, int] = {role: i for i, role in enumerate(Roles)}

    def create_signs(self):
        # Information preparation
        actions: List[Action] = extract_actions(self.text_info)
        clusters: List[Cluster] = extract_clusters(self.text_info)
        combine_actions_with_clusters(actions, clusters, self.text_info)
        resolve_pronouns(clusters)




