from isanlp import PipelineCommon
from isanlp.annotation_repr import CSentence
from isanlp.annotation import Event, TaggedSpan
import subprocess as sbp
import json
import sys
import os


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))



class ProcessorDeepSrlWrapper:
    def __init__(self, model_path, pidmodel_path):
        self._proc = sbp.Popen(['python2', get_script_dir() + '/interprocess_server.py', 
                                '--model', model_path, 
                                '--pidmodel', pidmodel_path], 
                               stdin=sbp.PIPE, 
                               stdout=sbp.PIPE)
        self._warmup()
        
    def __del__(self):
        self._proc.terminate()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, value, traceback):
        self._proc.terminate()
        
    def _warmup(self):
        self._process_json('["This", "is", "warmup"]')
        
    def _process_json(self, json_str):
        self._proc.stdin.write((json_str + '\n\n').encode('utf8'))
        self._proc.stdin.flush()

        result_str = b''
        while not result_str.endswith(b'\n\n'):
            if self._proc.poll() is not None:
                raise RuntimeError('Subprocess terminated, can no longer continue. The object should be recreated.')
                
            result_str += self._proc.stdout.readline()

        return result_str[:-2].decode('utf8')
        
    def __call__(self, tokens, sentences):
        sys.stderr.write('Processing input...\n')
        sys.stderr.flush()
        
        input_data = [[word.text for word in CSentence(tokens, sent)] for sent in sentences]
        result_str = self._process_json(json.dumps(input_data))
        result_json = json.loads(result_str)
        result = []
        for sent in result_json:
            result_sent = []
            for pred_arg in sent:
                result_sent.append(Event(pred = (pred_arg[0], pred_arg[0]), 
                                         args = [TaggedSpan(arg[0], arg[1], arg[2]) 
                                                 for arg in pred_arg[1] if arg[0] != 'V']))
            
            result.append(result_sent)
        
        return result


DEEP_SRL = PipelineCommon([(ProcessorDeepSrlWrapper("/src/deep_srl/resources/conll05_propid_model",
                                                    "/src/deep_srl/resources/conll05_model"), 
                           ['tokens', 'sentences'], 
                           {0 : 'srl'})],
                          name = 'default')
