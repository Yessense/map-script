import os
import logging
from typing import List, Tuple, Any

from src.script_extraction.graph.graph import Graph, show_graph
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info


def extract_all(files: List) -> List[Tuple[Any, Any]]:
    texts_info = extract_texts_info(files)


    logging.info("-" * 40)
    logging.info("Creating graphs")
    logging.info("-" * 40)

    graphs = []
    for filepath, text_info in zip(files, texts_info):
        logging.info(f"Processing {filepath}")
        V, E = Graph(text_info).get_graph()
        graphs.append((V, E))

    return graphs


def show_graphs(files: List[str], graphs: List[Tuple[Any, Any]]):
    for f, (V, E) in zip(files, graphs):
        show_graph(V, E, f)


def main():
    TEXT_FOLDER = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/"

    file_names = [os.path.splitext(os.path.basename(f))[0] + '.html'
                  for f in os.listdir(TEXT_FOLDER)
                  if os.path.isfile(os.path.join(TEXT_FOLDER, f))]
    files = [os.path.join(TEXT_FOLDER, f)
             for f in os.listdir(TEXT_FOLDER)
             if os.path.isfile(os.path.join(TEXT_FOLDER, f))]
    graphs = extract_all(files)

    show_graphs(file_names, graphs)
    print("DONE")




if __name__ == '__main__':
    main()
