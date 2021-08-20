"""
Add roles, clusters
"""
from typing import Dict, List, Any, Tuple

from src.script_extraction.text_preprocessing.resolve_phrases import resolve_phrases, get_trees_list
from src.script_extraction.text_preprocessing.words_object import Cluster, WordsObject, Position, Obj, Roles, POS
from src.text_info_cinema import create_text_info_cinema
from src.text_info_restaurant import create_text_info_restaurant


def get_sentences_bounds(text_info: Dict) -> List[Tuple[int, int]]:
    """
    Create bounds for each sentence
    (start word, end word + 1)
    Parameters
    ----------
    text_info: Dict

    Returns
    -------
    out: List[Tuple[int, int]

    """
    sentences_bounds: List[Tuple[int, int]] = []
    current_word_number = 0

    for sentence in text_info['sentences_info']:
        number_of_words_in_sentence = len(sentence['semantic_roles']['words'])
        sentences_bounds.append((current_word_number,
                                 current_word_number + number_of_words_in_sentence))
        current_word_number += number_of_words_in_sentence
    return sentences_bounds


def extract_clusters(text_info: Dict) -> List[Cluster]:
    """
    Collect info about clusters in one list
    Parameters
    ----------
    text_info: Dict

    Returns
    -------
    out: List[Cluster]

    """
    clusters: List[Cluster] = []

    sentences_bounds = get_sentences_bounds(text_info)

    for cluster_index, cluster_info in enumerate(text_info['coreferences']['clusters']):
        cluster = Cluster()
        for entry in cluster_info:
            for sentence_number, sentence in enumerate(text_info['sentences_info']):
                if (entry[1] < sentences_bounds[sentence_number][1]
                        and entry[0] >= sentences_bounds[sentence_number][0]):
                    start_word = entry[0] - sentences_bounds[sentence_number][0]
                    end_word = entry[1] - sentences_bounds[sentence_number][0] + 1

                    position = Position(sentence_number=sentence_number,
                                        start_word=start_word,
                                        end_word=end_word)

                    obj = Obj(position=position,
                              text=" ".join(
                                  text_info['coreferences']['document'][entry[0]:entry[1] + 1]),
                              arg_type=Roles.NAMED_GROUP)
                    obj.set_part_of_speech(sentences_info=text_info['sentences_info'])
                    if obj.is_accepted:
                        obj.position.set_symbols_bounds(sentence['semantic_roles']['words'], obj.text)
                        cluster.add_cluster_obj(obj)
        clusters.append(cluster)

    trees_list = get_trees_list(text_info)
    resolve_phrases(clusters, trees_list, text_info)
    return clusters


PRON_I = {"i", "me", "my", "mine", "myself"}
PRON_I_REPLACE = "person"

PRON_WE = {"we", "us", "our", "ours", "ourselves"}
PRON_WE_REPLACE = "people"

PRON_IT = {"it", "its", "itself"}
PRON_IT_REPLACE = "object"

def replace_objects_in_cluster(cluster: Cluster, replace_word: str):
    cluster.real_objects = cluster.real_objects[:1]
    cluster.real_objects[0].text = replace_word
    cluster.real_objects[0].pos = POS.NOUN


def resolve_pronouns(clusters: List[Cluster]):
    # If there is NOUN, remove all
    for cluster in clusters:
        delete_prons: bool = False
        for real_obj in cluster.real_objects:
            if real_obj.pos is POS.NOUN or real_obj.pos is POS.VERB:
                delete_prons = True
        if delete_prons:
            cluster.real_objects = [real_obj for real_obj in cluster.real_objects if real_obj.pos is POS.PRON]
            return

    for cluster in clusters:
        need_to_it_replace = True
        for real_obj in cluster.real_objects:
            if real_obj.lemma in PRON_I:
                need_to_replace = False
                replace_objects_in_cluster(cluster, replace_word=PRON_I_REPLACE)
            elif real_obj.lemma in PRON_WE:
                need_to_replace = False
                replace_objects_in_cluster(cluster, replace_word=PRON_WE_REPLACE)
        if need_to_it_replace:
            replace_objects_in_cluster(cluster, replace_word=PRON_IT_REPLACE)




def example_usage() -> None:
    # text_info
    text_info = create_text_info_restaurant()
    clusters = extract_clusters(text_info)
    # semantic_roles with coreferences
    print("DONE")


if __name__ == '__main__':
    example_usage()
