from typing import Dict, Any, List

from mapcore.swm.src.components.semnet import Sign
from mapcore.swm.src.components.semnet import CausalMatrix

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_actions
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from src.script_extraction.text_preprocessing.role import Role


def create_script_sign(text_info: Dict[str, Any]):

    script_name = "Script"
    S = Sign(script_name)

    actions_signs = {}
    roles_signs = {}
    objects_signs = {}
    images_signs = {}

    significances: Dict[str, CausalMatrix] = {}
    significances[script_name] = S.add_significance()





def example_usage():

    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    create_script_sign(text_info)
    print("Done")



if __name__ == '__main__':
    example_usage()