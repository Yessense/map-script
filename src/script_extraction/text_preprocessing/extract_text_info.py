import logging
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


import nltk.data

logger = logging.getLogger('dev')
logger.setLevel(logging.WARNING)


def get_predictors():
    """
    Retrieves all needed  predictors:
    coreference
    open information
    dependency
    semantic role
    Returns
    -------
    out : dict of dict
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
    logger.info("Start loading predictors.")
    logger.setLevel(logging.WARNING)

    for name in predictors:
        logger.setLevel(logging.INFO)
        logger.info(f'Loading: {name}...')
        logger.setLevel(logging.WARNING)

        predictors[name]['predictor'] = Predictor.from_path(predictors[name]['path'])

    logger.setLevel(logging.INFO)
    logger.INFO("Predictors loaded.")

    return predictors

# text file name
# TODO: make files list
FILE_NAME = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt"

# using nltk for splitting to sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# coreference resolution
coreference_finder = Predictor.from_path()

# various info
open_information_extractor = Predictor.from_path()
dependency_parser = Predictor.from_path()
semantic_role_labeler = Predictor.from_path()

# --------------------------------------------------------------
# PARSING
# --------------------------------------------------------------

logger.setLevel(logging.INFO)

with open(FILE_NAME) as f:
    text = f.read()

logger.info("File is loaded.")

text_info = {}

# finding coreference

coreferences = coreference_finder.predict(document=text)
text_info['coreferences'] = coreferences

logger.info("Predicting coreferences")

# extract information
tokenized_text = tokenizer.tokenize(text)
predictors = {"open information": open_information_extractor,
              "dependancy": dependency_parser,
              "semantic role": semantic_role_labeler}
sentences_info = []
for sentence_index, sentence in enumerate(tokenized_text):
    sentence_info = {}
    for predictor_name, predictor in predictors.items():
        parsed_data = predictor.predict(sentence)
        sentence_info[predictor_name] = parsed_data
    sentences_info.append(sentence_info)

text_info['sentences_info'] = sentence_info
print("Done")
text_info['coreferences']['clusters_words'] = [[" ".join(text_info['coreferences']['document'][words[0]:words[1] + 1])
                                                for words in cluster]
                                               for cluster in text_info['coreferences']['clusters']]
