from nltk.corpus import propbank
pb_instances = propbank.instances()

inst = pb_instances[54]


tree = inst.tree

from nltk.corpus import treebank
assert tree == treebank.parsed_sents(inst.fileid)[inst.sentnum]
predicate = inst.predicate.select(tree)



expose_01 = propbank.roleset('run.04')
for role in expose_01.findall("roles/role"):
    print(role.attrib['n'], role.attrib['descr'])

verbs = propbank.verbs()[1000]
print("Done")