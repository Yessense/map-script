from processor_deep_srl import ProcessorDeepSrl

import argparse
from itertools import izip
import json
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',
                      type=str,
                      default='',
                      required=True,
                      help='SRL Model path.')

    parser.add_argument('--pidmodel',
                      type=str,
                      default='',
                      help='Predicate identfication model path.')

    args = parser.parse_args()
    
    sys.stderr.write('Starting parser...\n')
    proc = ProcessorDeepSrl(args.model, args.pidmodel)
    sys.stderr.write('Parser started.\n')
    
    while(True):
        input_str = ''
        while not input_str.endswith('\n\n'):
            input_str += sys.stdin.readline()
            sys.stderr.flush()
        
        sys.stderr.write('Starting analysis...\n')
        sys.stderr.flush()
        
        res = proc(json.loads(input_str.decode('utf8')))
        
        sys.stderr.write('Done.\n')
        sys.stderr.flush()
        
        sys.stdout.write((json.dumps(res) + '\n\n').encode('utf8'))
        sys.stdout.flush()
        