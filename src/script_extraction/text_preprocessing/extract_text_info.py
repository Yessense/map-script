from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk.data

file_name = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/restaurant.txt"

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

with open(file_name) as f:
    text = f.read()

tokenized_text = tokenizer.tokenize(text)

open_information_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
dependency_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
semantic_role_labeler = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

predictors = {"open information": open_information_extractor,
              "dependancy": dependency_parser,
              "semantic role": semantic_role_labeler}
text_info = {}
for sentence_index, sentence in enumerate(tokenized_text):
    sentence_info = {}
    for predictor_name, predictor in predictors.items():
        parsed_data = predictor.predict(sentence)
        sentence_info[predictor_name] = parsed_data
    text_info[sentence_index] = sentence_info

print("Done")
