from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote


def get_semantic_roles(text):
    HOST = 'localhost'
    proc_morph = ProcessorRemote(host=HOST,
                                 port=3333,
                                 pipeline_name='default')
    proc_sem = ProcessorRemote(host=HOST,
                               port=3334,
                               pipeline_name='default')

    pipeline = PipelineCommon([(proc_morph, ['text'],
                                {'tokens': 'tokens',
                                 'sentences': 'sentences'}),
                               (proc_sem, ['tokens', 'sentences'], {'srl': 'srl'})])
    # try:
    analysis_result = pipeline(text)
    # except:
    #     print("Check docker container")
    #     return None
    return analysis_result


def example_usage():
    filename = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/cinema.txt'
    with open(filename, 'r') as f:
        text = f.read()
    roles = get_semantic_roles(text)
    print("Done")


if __name__ == '__main__':
    example_usage()
