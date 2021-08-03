from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.wsd import lesk


sent = ['We', 'choose', 'movie', 'for', 'the', 'family', ',', 'we', 'need', 'something', 'pleasant', ',', 'amusing', 'and', 'funny', '.']

for i,j in enumerate(wn.synsets('family')):
    print('Meaning', i, 'NLTK ID', j.name())
    print('Definition:', j.definition())
    print('Hypernyms:', ', '.join(list(chain(*[l.lemma_names() for l in j.hypernyms()]))))

ss = lesk(sent, 'family')
lemmas = set()
for hyper in ss.hypernyms():
    lemmas.update(hyper.lemma_names())
print("Done")
