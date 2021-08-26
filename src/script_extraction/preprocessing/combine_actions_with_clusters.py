from typing import List, Dict, Tuple, Union, Any

from src.script_extraction.preprocessing.extract_clusters import extract_clusters, resolve_pronouns
from src.script_extraction.preprocessing.extract_semantic_roles import extract_actions
from src.script_extraction.preprocessing.words_object import Action, Cluster, WordsObject, Obj
from src.script_extraction.samples.text_info.text_info_restaurant import create_text_info_restaurant


def add_meanings(actions: List[Action], clusters: List[Cluster], text_info: Dict[str, Any]):
    """
    Resolve semantic meaning for each word

    Parameters
    ----------
    actions
    clusters
    text_info

    Returns
    -------

    """
    for action in actions:
        action.set_meaning(text_info)
        for obj in action.objects:
            obj.set_meaning(text_info)
            for image in obj.images:
                image.set_meaning(text_info)
    for cluster in clusters:
        for obj in cluster.objects:
            obj.set_meaning(text_info)


def create_clusters_dict(clusters: List[Cluster]) -> Dict[Tuple[int, int, int], Cluster]:
    """
    Create dict {word index: Cluster}
    Parameters
    ----------
    clusters

    Returns
    -------

    """
    clusters_dict: Dict[Tuple[int, int, int], Cluster] = dict()
    for cluster in clusters:
        for obj in cluster.objects:
            clusters_dict[obj.index] = cluster
    return clusters_dict


def combine_actions_with_clusters(actions: List[Action],
                                  clusters: List[Cluster],
                                  text_info: Dict) -> List[Action]:
    """
    Add cluster field for each real object in text
    """
    add_meanings(actions, clusters, text_info)
    clusters_dict: Dict[Tuple[int, int, int], Cluster] = create_clusters_dict(clusters)

    def process_real_obj(obj: Union[WordsObject, Obj, Action]) -> None:
        if obj.index in clusters_dict:
            clusters_dict[obj.index].add_real_obj(obj)
            obj.cluster = clusters_dict[obj.index]

    for action in actions:
        process_real_obj(action)
        for obj in action.objects:
            process_real_obj(obj)
            for image in obj.images:
                process_real_obj(image)
    return actions


def example_usage() -> None:
    # _text_info
    text_info = create_text_info_restaurant()

    actions = extract_actions(text_info)
    clusters = extract_clusters(text_info)
    combine_actions_with_clusters(actions, clusters, text_info)
    resolve_pronouns(clusters)
    print("DONE")


if __name__ == '__main__':
    example_usage()
