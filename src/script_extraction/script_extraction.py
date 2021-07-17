import os
import logging
from typing import List, Tuple, Any, Dict

from src.script_extraction.graph.graph import Graph, show_graph, add_hyponyms_hypernyms_synonyms, Vertex, Edge
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info


def extract_all(files: List) -> List[Tuple[Dict[Any, Vertex], Dict[Any, Edge]]]:
    file_names = [os.path.splitext(os.path.basename(f))[0] + ".html"
                  for f in files]
    texts_info = extract_texts_info(files)


    logging.info("-" * 40)
    logging.info("Creating graphs")
    logging.info("-" * 40)

    graphs = []
    for filepath, text_info in zip(files, texts_info):
        logging.info(f"Creating graph {filepath}")
        graphs.append(Graph(text_info).get_graph())

    for V, E in graphs:
        add_hyponyms_hypernyms_synonyms(V)

    for f, (V, E) in zip(file_names, graphs):
        show_graph(V, E, f)

    return graphs




def main():
    TEXT_FOLDER = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/"

    files = [os.path.join(TEXT_FOLDER, f)
             for f in os.listdir(TEXT_FOLDER)
             if os.path.isfile(os.path.join(TEXT_FOLDER, f))]

    graphs = extract_all(files)


    print("DONE")




if __name__ == '__main__':
    main()
