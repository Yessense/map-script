from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info


class graph:
    def __init__(self, text_info):

        for sentence_info in text_info['sentences_info']:
            self.trees.append(graph.get_tree(sentence_info))
        self.G = graph.get_graph_from_trees(trees=self.trees)




def example_usage():
    pass


if __name__ == '__main__':
    example_usage()
