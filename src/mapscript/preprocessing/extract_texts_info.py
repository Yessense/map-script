import logging
import os
import os.path
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any
from allennlp.predictors.predictor import Predictor
import nltk.data # type: ignore
import hashlib # type: ignore

logging.getLogger("predictor").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("filelock").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt='%H:%M:%S')


def get_predictors() -> Dict[str, Dict[str, Any]]:
    """
    Load or unzip all predictors:

    - coreference
    - dependency
    - semantic role
    - tokenizer

    Returns
    -------
    out: Dict[str, Dict[str, Any]]
        {'name': {'path': url,
                  'predictor': Predictor}}
    """
    predictors = {
        "dependency": {
            'path': "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"},
        "semantic_roles": {
            'path': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"},
        "coreferences": {
            'path': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"},
    }

    logging.info("Loading predictors")

    for name in predictors:
        predictors[name]['predictor'] = Predictor.from_path(predictors[name]['path'])
        logging.info(f'{name} is loaded')


    # using nltk for splitting to sentences
    predictors['tokenizer'] = {'path': 'tokenizers/punkt/english.pickle'}
    predictors['tokenizer']['predictor'] = nltk.data.load(predictors['tokenizer']['path'])
    logging.info(f'nltk tokenizer is loaded')

    logging.info("All predictors are loaded.")
    return predictors


def get_text_info(filename, predictors):
    """
    Retrieves info from one file:
    Coreferences in all text

    semantic role: verbs and their arguments
    dependency: full tree

    Parameters
    ----------
    filename: str
    predictors: dict


    Returns
    -------

    """
    predictors_names = ["semantic_roles", "dependency"]

    with open(filename) as f:
        text = f.read()

    logging.info(f" {filename} is loaded.")

    text_info = {}

    logging.info("Predicting coreferences")
    # finding coreference
    coreferences = predictors['coreferences']['predictor'].predict(document=text)
    text_info['coreferences'] = coreferences

    # extract information
    tokenized_text = predictors['tokenizer']['predictor'].tokenize(text)

    logging.info("Predicting sentences info")
    sentences_info = []
    for sentence_index, sentence in enumerate(tokenized_text):
        sentence_info = {}
        for predictor_name in predictors_names:
            parsed_data = predictors[predictor_name]['predictor'].predict(sentence)
            sentence_info[predictor_name] = parsed_data
        sentences_info.append(sentence_info)

    text_info['sentences_info'] = sentences_info
    return text_info


def extract_texts_info(files=None, processed_cache: str = None) -> List[Dict]:
    """
    Retrieves info from each file
    Parameters
    ----------
    processed_cache: str
    files: list of str

    Returns
    -------
    out: list
        list of text info
    """

    if files is None:
        files = []
    if not len(files):
        return []
    if processed_cache is None:
        processed_cache = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), ".processed_cache")
    if not os.path.exists(processed_cache):
        os.makedirs(processed_cache)

    predictors = None
    logging.info("-" * 40)
    logging.info("Processing files")
    logging.info("-" * 40)
    texts_info = []

    saved_files = [os.path.basename(f) for f in os.listdir(processed_cache)]
    for filepath in files:
        h = hashlib.sha1(filepath.encode()).hexdigest()
        if h in saved_files:
            with open(os.path.join(processed_cache, h), 'rb') as f:
                info = pickle.load(f)
        else:
            if predictors is None:
                predictors = get_predictors()
            info = get_text_info(filepath, predictors)
            with open(os.path.join(processed_cache, h), 'wb') as f:
                pickle.dump(info, f)
        texts_info.append(info)
    logging.info("All files processed")
    return texts_info


def example_usage() -> None:
    def get_project_root() -> Path:
        return Path(__file__).parent.parent.parent.parent
    file_name: str = os.path.join(get_project_root(), "example_usage/", "my_text.txt")
    files = [file_name]
    if __name__ == "__main__":
        processed_cache = os.path.join(get_project_root(), ".processed_cache")
    texts_info = extract_texts_info(files, processed_cache=processed_cache)

    print("Done")


if __name__ == "__main__":

    example_usage()
