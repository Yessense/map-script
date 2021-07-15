import os
from typing import List

from src.script_extraction.graph.graph import Graph
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info


def extract_all(files: List) -> List[Graph]:
    texts_info = extract_texts_info(files)













def main():
    TEXT_FOLDER = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/"
    GRAPHS_FOLDER = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/graphs/"

    file_names =[os.path.basename(f) for f in os.listdir(TEXT_FOLDER)
                 if os.path.isfile(os.path.join(TEXT_FOLDER, f))]
    files = [os.path.join(TEXT_FOLDER, f)
             for f in os.listdir(TEXT_FOLDER)
             if os.path.isfile(os.path.join(TEXT_FOLDER, f))]

    extract_all(files)

if __name__ == '__main__':
    main()

