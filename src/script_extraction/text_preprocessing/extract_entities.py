from src.script_extraction.text_preprocessing.cluster import Cluster, Element
from tests.get_info import get_text_info


def add_sentences_bounds(text_info):
    number_of_previous_words = 0

    for sentence in text_info['sentences_info']:
        number_of_words_in_sentence = len(sentence['dependency']['words'])
        sentence['sentence_bounds'] = (
            number_of_previous_words, number_of_previous_words + number_of_words_in_sentence)
        number_of_previous_words += number_of_words_in_sentence


def create_coreferences_clusters(text_info):
    add_sentences_bounds(text_info)
    clusters = []
    for cluster_info in text_info['coreferences']['clusters']:
        cluster = Cluster()
        for entry in cluster_info:
            for sentence_number, sentence in enumerate(text_info['sentences_info']):
                if entry[1] < sentence['sentence_bounds'][1] and entry[0] >= sentence['sentence_bounds'][0]:
                    start_word_number = entry[0] - sentence['sentence_bounds'][0]
                    end_word_number = entry[1] - sentence['sentence_bounds'][0]

                    element = Element(sentence_number=sentence_number,
                                      word_spans=(start_word_number, end_word_number),
                                      string=" ".join(text_info['coreferences']['document'][entry[0]:entry[1] + 1]))

                    cluster.add_element(element)
        clusters.append(cluster)
    text_info['coreferences']['clusters_info'] = clusters



def example_usage():
    # text_info
    text_info = get_text_info()


    # create coreferences clusters
    text_info_with_coreferences_clusters = get_text_info()
    create_coreferences_clusters(text_info_with_coreferences_clusters)

    # semantic_roles with coreferences
    print("DONE")


if __name__ == '__main__':
    example_usage()
