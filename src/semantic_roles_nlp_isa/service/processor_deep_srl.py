from neural_srl.shared import *
from neural_srl.shared.constants import *
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.inference import *
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.shared.evaluation import SRLEvaluator
from neural_srl.shared.io_utils import bio_to_spans
from neural_srl.shared.reader import string_sequence_to_ids
from neural_srl.shared.scores_pb2 import *
from neural_srl.shared.tensor_pb2 import *
from neural_srl.theano.tagger import BiLSTMTaggerModel
from neural_srl.theano.util import floatX

from itertools import izip
import numpy
import os
import sys
import theano


def load_model(model_path, model_type):
    config = configuration.get_config(os.path.join(model_path, 'config'))
    # Load word and tag dictionary
    word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
    label_dict = Dictionary()
    word_dict.load(os.path.join(model_path, 'word_dict'))
    label_dict.load(os.path.join(model_path, 'label_dict'))
    data = TaggerData(config, [], [], word_dict, label_dict, None, None)

    if model_type == 'srl':
        test_sentences, emb_inits, emb_shapes = reader.get_srl_test_data(
            None, config, data.word_dict, data.label_dict, False)
    else:
        test_sentences, emb_inits, emb_shapes = reader.get_postag_test_data(
            None, config, data.word_dict, data.label_dict, False)
  
    data.embedding_shapes = emb_shapes
    data.embeddings = emb_inits
    model = BiLSTMTaggerModel(data, config=config, fast_predict=True)
    model.load(os.path.join(model_path, 'model.npz'))
    return model, data


class ProcessorDeepSrl:
    def __init__(self, pidmodel_path, model_path):
        self.pid_model, self.pid_data = load_model(pidmodel_path, 'propid')
        self.srl_model, self.srl_data = load_model(model_path, 'srl')
        
        self.transition_params = get_transition_params(self.srl_data.label_dict.idx2str)

        self.pid_pred_function = self.pid_model.get_distribution_function()
        self.srl_pred_function = self.srl_model.get_distribution_function()
    
    def __call__(self, sentences):
        srl_sent = []
        for sent in sentences:
            num_tokens = len(sent)
            s0 = string_sequence_to_ids(sent, self.pid_data.word_dict, True)
            s0 = [e if e < 33397 else 0 for e in s0] # TODO: Ugly FIX There is a mismatch in a Conll2005 dictionary and a model 
            l0 = [0 for _ in s0]
            x, _, _, weights = self.pid_data.get_test_data([(s0, l0)], batch_size=None)
            pid_pred, scores0 = self.pid_pred_function(x, weights)

            s1 = []
            predicates = []
            for i,p in enumerate(pid_pred[0]):
                if self.pid_data.label_dict.idx2str[p] == 'V':
                    predicates.append(i)
                    feats = [1 if j == i else 0 for j in range(num_tokens)]
                    s1.append((s0, feats, l0))

            if len(s1) == 0:
                srl_sent.append([])
                continue

            # Semantic role labeling.
            x, _, _, weights = self.srl_data.get_test_data(s1, batch_size=None)
            srl_pred, scores = self.srl_pred_function(x, weights)
            #print srl_pred 

            arguments = []
            for i, sc in enumerate(scores):
                viterbi_pred, _ = viterbi_decode(sc, self.transition_params)
                arg_spans = bio_to_spans(viterbi_pred, self.srl_data.label_dict)
                arguments.append(arg_spans)
    
            # Print human-readable results.
            srl_sent.append(list(izip(predicates, arguments)))
       
        return srl_sent
