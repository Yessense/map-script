import logging
import os
import pickle
from typing import List, Dict
from allennlp.predictors.predictor import Predictor
import nltk.data # type: ignore
import hashlib # type: ignore

logging.getLogger("predictor").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("filelock").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt='%H:%M:%S')


def get_predictors():
    """
    Retrieves all needed  predictors, load or unzip:
    coreference
    dependency
    semantic role

    Returns
    -------
    out : dict of {str: str}
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

    logging.info("-" * 40)
    logging.info("Loading predictors")
    logging.info("-" * 40)

    for name in predictors:
        logging.info(f'Loading: {name}.')

        predictors[name]['predictor'] = Predictor.from_path(predictors[name]['path'])

    logging.info(f'Loading: nltk tokenizer.')

    # using nltk for splitting to sentences
    predictors['tokenizer'] = {'path': 'tokenizers/punkt/english.pickle'}
    predictors['tokenizer']['predictor'] = nltk.data.load(predictors['tokenizer']['path'])

    logging.info("Predictors loaded.")
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

    logging.info(f"File {filename} is loaded.")

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


def extract_texts_info(files=None, saved_files_dir: str = None) -> List[Dict]:
    """
    Retrieves info from each file
    Parameters
    ----------
    saved_files_dir: str
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
    if saved_files_dir is None:
        saved_files_dir = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/saved/"

    predictors = None
    logging.info("-" * 40)
    logging.info("Processing files")
    logging.info("-" * 40)
    texts_info = []

    saved_files = [os.path.basename(f) for f in os.listdir(saved_files_dir)]
    for filepath in files:
        h = hashlib.sha1(filepath.encode()).hexdigest()
        if h in saved_files:
            with open(os.path.join(saved_files_dir, h), 'rb') as f:
                info = pickle.load(f)
        else:
            if predictors is None:
                predictors = get_predictors()
            info = get_text_info(filepath, predictors)
            with open(os.path.join(saved_files_dir, h), 'wb') as f:
                pickle.dump(info, f)
        texts_info.append(info)
    logging.info("All files processed")
    return texts_info


def example_usage() -> None:
    FILE_NAME = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt"
    files = [FILE_NAME]
    texts_info = extract_texts_info(files)

    print("Done")


if __name__ == "__main__":
    example_usage()
