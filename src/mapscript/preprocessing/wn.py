from typing import List, Tuple, Optional, Any, Dict

from nltk.corpus import wordnet as wn
from itertools import chain

from nltk.corpus.reader import Synset
from nltk.wsd import lesk


def get_meaning(sentence: List[str], lemma: str, pos: str) -> Tuple[int, int]:
    synsets: List[Synset] = wn.synsets(lemma)
    # word not represented in dict
    if not len(synsets):
        return 1, 0
    # word has only one meaning
    elif len(synsets) == 1:
        return 1, 0

    synset: Synset = lesk(context_sentence=sentence,
                          ambiguous_word=lemma,
                          pos=pos,
                          synsets=synsets)
    if synset is None:
        # check for 's' pos
        if pos == 'a':
            synset = lesk(context_sentence=sentence,
                          ambiguous_word=lemma,
                          pos=pos,
                          synsets=synsets)
            if synset is None:
                return -1, -1
        else:
            return -1, -1

    name = synset.name()

    for i, ss in enumerate(synsets):
        if ss.name() == name:
            return len(synsets), i

    return -1, -1


def get_synsets(lemma: str) -> Dict[str, Synset]:
    ss = wn.synsets(lemma)
    return {synset.name(): synset for synset in ss}


def get_hypernyms(synset: Synset) -> Dict[str, Synset]:
    return {ss.name(): ss for ss in synset.hypernyms()}


def get_synset_number(lemma: str, synset_name) -> Tuple[int, int]:
    ss: List[Synset] = wn.synsets(lemma)
    for i, synset in enumerate(ss):
        if synset.name() == synset_name:
            return i, len(ss)
    return -1, -1


def example_usage():
    sent = ['We', 'choose', 'movie', 'for', 'the', 'family', ',', 'we', 'need', 'something', 'pleasant', ',', 'amusing',
            'and', 'funny', '.']

    ss = get_synsets("man")
    # stack overflow
    for i, j in enumerate(wn.synsets('individual')):
        print('Meaning', i, 'NLTK ID', j.name())
        print('Definition:', j.definition())
        z = j.hypernyms()
        x = j.hyponyms()
        print('Hypernyms:', ', '.join(list(chain(*[l.lemma_names() for l in j.hypernyms()]))))

    ss = lesk(sent, 'family')

    lemmas = set()

    for hyper in ss.hypernyms():
        lemmas.update(hyper.lemma_names())
    sent = "I go to school".split()
    test = wn.synsets('go', pos='v')
    ss = lesk(sent, 'go', pos='v')

    result = get_meaning(sent, 'go', 'v')

    sent = ['Even', 'though', 'it', 'must', 'have', 'been', 'very', 'time', '-', 'consuming', 'to', 'prepare', ',',
            'it', 'was', 'a', 'delight', 'to', 'see', ',', 'and', 'I', 'had', 'a', 'second', 'helping', '.']
    lemma = 'must'
    synsets = wn.synsets('i')

    result = lesk(sent, lemma, synsets=synsets)
    print("Done")


if __name__ == '__main__':
    example_usage()
