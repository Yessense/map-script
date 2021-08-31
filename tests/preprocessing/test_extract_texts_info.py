from src.mapscript.preprocessing.extract_texts_info import get_predictors, get_text_info


class Test_get_predictors:
    def test_get_predictors(self):
        predictor_names = ['dependency', 'semantic_roles', 'coreferences', 'tokenizer']
        predictors = get_predictors()

        assert len(predictors) == 4
        for name in predictor_names:
            assert name in predictors
            assert predictors[name]['predictor'] is not None
            assert len(predictors[name]['path'])


class Test_get_text_info:
    def test_get_text_info(self):
        predictors = get_predictors()
        filename = 'cinema.txt'
        text_info = get_text_info(filename, predictors)
        assert len(text_info) == 2

