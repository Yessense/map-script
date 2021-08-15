from typing import Any, Dict, List, Union

from mapcore.swm.src.components.semnet import Sign

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions, \
    combine_actions_with_clusters
from src.script_extraction.text_preprocessing.words_object import Action, Cluster, WordsObject, Obj
from src.text_info_restaurant import create_text_info_restaurant


def add_new_sign(script: Dict[str, Sign], obj: Union[WordsObject, Obj, Action]) -> Dict[str, Sign]:
    name = obj.lemma

    if name not in script:
        sign = Sign(name)

        for i in range(obj.synsets_len):
            sign.add_significance(pm=None)

    return script


def extract_script(text_info: Dict[str, Any]):
    actions: List[Action] = extract_actions(text_info)
    clusters: List[Cluster] = extract_clusters(text_info)
    combine_actions_with_clusters(actions, clusters, text_info)

    script: Dict[str, Sign] = dict()
    for action in actions:
        add_new_sign()

    return script


def main():
    text_info = create_text_info_restaurant()

    script = extract_script(text_info)

    print("DONE")


if __name__ == '__main__':
    main()
