from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open("/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt") as f:
    text = f.read()

tokenized_text = tokenizer.tokenize(text)

open_information_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
dependency_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
semantic_role_labeler = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")