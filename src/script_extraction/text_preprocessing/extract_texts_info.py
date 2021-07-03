import logging
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

import nltk.data

logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)


def get_predictors():
    """
    Retrieves all needed  predictors, load or unzip:
    coreference
    open information
    dependency
    semantic role

    Returns
    -------
    out : dict of {str: str}
        {'name': {'path': url,
                  'predictor': Predictor}}
    """
    predictors = {
        "open information": {
            'path': "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"},
        "dependency": {
            'path': "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"},
        "semantic role": {
            'path': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"},
        "coreference": {
            'path': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"},
    }

    logger = logging.getLogger('dev')
    logging.info("Start loading predictors.")
    logger.setLevel(logging.WARNING)

    for name in predictors:
        logger.setLevel(logging.INFO)
        logging.info(f'Loading: {name}...')
        logger.setLevel(logging.WARNING)

        predictors[name]['predictor'] = Predictor.from_path(predictors[name]['path'])

    logger.setLevel(logging.INFO)
    logging.info(f'Loading: tokenizer...')
    logger.setLevel(logging.WARNING)

    # using nltk for splitting to sentences
    predictors['tokenizer'] = {'path': 'tokenizers/punkt/english.pickle'}
    predictors['tokenizer']['predictor'] = nltk.data.load(predictors['tokenizer']['path'])

    logger.setLevel(logging.INFO)
    logging.info("Predictors loaded.")

    return predictors


def get_text_info(filename, predictors):
    with open(filename) as f:
        text = f.read()

    logging.info("File is loaded.")

    text_info = {}

    # finding coreference
    coreferences = predictors['coreference']['predictor'].predict(document=text)
    text_info['coreferences'] = coreferences

    logging.info("Predicting coreferences")

    # extract information
    tokenized_text = predictors['tokenizer']['predictor'].tokenize(text)

    predictors_names = ["open information", "dependency", "semantic role"]
    sentences_info = []
    for sentence_index, sentence in enumerate(tokenized_text):
        sentence_info = {}
        for predictor_name in predictors_names:
            parsed_data = predictors[predictor_name]['predictor'].predict(sentence)
            sentence_info[predictor_name] = parsed_data
        sentences_info.append(sentence_info)

    text_info['sentences_info'] = sentences_info
    return text_info


def extract_texts_info(files=[]):

    if not len(files):
        return []

    predictors = get_predictors()

    texts_info = []
    for filepath in files:
        info = get_text_info(filepath, predictors)
        texts_info.append(info)
    return texts_info


def example_usage():
    FILE_NAME = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt"
    files = [FILE_NAME]
    texts_info = extract_texts_info(files)

    print("Done")


if __name__ == "__main__":
    example_usage()
