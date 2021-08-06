"""
Add roles, clusters
"""
from typing import Dict, List, Any, Tuple

from src.script_extraction.text_preprocessing.words_object import Cluster, WordsObject, Position
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
                    words_object = WordsObject(position=position,
                                               text=" ".join(
                                                   text_info['coreferences']['document'][entry[0]:entry[1] + 1]))
                    words_object.set_part_of_speech(sentences_info=text_info['sentences_info'])

                    cluster.add_words_object(words_object)
        clusters.append(cluster)
    return clusters


def example_usage() -> None:
    # text_info
    text_info = create_text_info_restaurant()
    clusters = extract_clusters(text_info)

    # semantic_roles with coreferences
    print("DONE")


if __name__ == '__main__':
    example_usage()
