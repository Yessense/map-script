from typing import Dict, Any, List

from mapcore.swm.src.components.semnet import Sign
from mapcore.swm.src.components.semnet import CausalMatrix

from src.script_extraction.text_preprocessing.extract_clusters import extract_clusters
from src.script_extraction.text_preprocessing.extract_semantic_roles import extract_semantic_roles
from src.script_extraction.text_preprocessing.extract_texts_info import extract_texts_info
from src.script_extraction.text_preprocessing.role import Role


def create_script_sign(text_info: Dict[str, Any]):

    script_name = "Script"
    S = Sign(script_name)

    actions_signs = {}
    roles_signs = {}
    objects_signs = {}
    char_signs = {}
    significances: Dict[str, CausalMatrix] = {}
    significances[script_name] = S.add_significance()

    # extract clusters
    clusters, elements_dict = extract_clusters(text_info)
    parts_of_speech = [sentence_info['dependency']['pos'] for sentence_info in text_info['sentences_info']]

    # extract roles
    verbs: List[Role] = []
    for sentence_number, sentence_info in enumerate(text_info['sentences_info']):
        verbs += extract_semantic_roles(sentence_info, sentence_number)

    for verb in verbs:
        if verb.text not in actions_signs:
            actions_signs[verb.text] = Sign(verb.text)
        significances[verb.text] = actions_signs[verb.text].add_significance()
        connector = significances[script_name].add_feature(significances[verb.text])
        actions_signs[verb.text].add_out_significance(connector)

        for role in verb.roles:
            if role.text not in roles_signs:
                roles_signs[role.text] = Sign(role.text)
                significances[role.text] = roles_signs[role.text].add_significance()
            connector = significances[verb.text].add_feature(significances[role.text], zero_out=True)
            roles_signs[role.text].add_out_significance(connector)



    print("Done")



def example_usage():

    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    text_info = extract_texts_info([filename])[0]

    create_script_sign(text_info)
    print("Done")



if __name__ == '__main__':
    example_usage()