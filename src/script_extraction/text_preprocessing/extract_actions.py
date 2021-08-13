from typing import Any, Dict, Set, List


from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions
from src.script_extraction.text_preprocessing.words_object import Action, Cluster
from src.text_info_cinema import create_text_info_cinema


def combine_actions_with_clusters(actions: List[Action], clusters: List[Cluster], text_info: Dict) -> List[Action]:

    for action in actions:
        action.set_meaning(text_info)
        for obj in action.objects:
            obj.set_meaning(text_info)
            for image in obj.images:
                image.set_meaning(text_info)
    return script


def main():
    text_info = create_text_info_cinema()
    script = create_sript(text_info)
    print("Done")

if __name__ == '__main__':
    main()
