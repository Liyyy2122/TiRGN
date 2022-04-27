# -*- coding: utf-8 -*-
import os

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):        
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index

entity2id, id2entity = load_index(os.path.join('entity2id.txt'))
print(list(entity2id.keys())[:10])
relation2id, id2relation = load_index(os.path.join('relation2id.txt'))
# assert (1==0 )
count = 0
count1 = 0
word_list = set()
for entity_str in entity2id.keys():
    if "(" in entity_str and ")" in entity_str:
        count += 1
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()
        w2 = entity_str[begin+1: end]
        if w2 not in entity2id.keys():
            # print(w2)
            count1 += 1
        loc, typ = w2.split('@')
        if loc !='':
            loc = loc.strip().split(' ')
            for loc_ in loc:
                word_list.add(loc_)
        # print(loc, typ)
        if typ !='':
            typ = typ.strip().split(' ')
            for typ_ in typ:
                word_list.add(typ_)
        word_list.add(w1)
        # word_list.add(w1)
        # word_list.add(w2)
    else:
        print("no () ", entity_str)
        assert (1==0)
        word_list.add(entity_str)

num_word = len(word_list)

word2id = {word: id for id, word in enumerate(word_list)}
id2word = {id: word for id, word in enumerate(word_list)}
# print(word2id)
# print(id2word)

print(len(word2id.keys()))
print(float(count)/len(entity2id.keys()))
print(float(count1)/float(count))

with open("word2id.txt", "w") as f:
    for word in word2id.keys():
        f.write(word + "\t" + str(word2id[word])+'\n')

eid2wid = []
for id in range(len(id2entity.keys())):
    entity_str = id2entity[str(id)]
    if "(" in entity_str and ")" in entity_str:
        count += 1
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()
        w2 = entity_str[begin+1: end]
        loc, typ = w2.split('@')
        if loc !='':
            loc = loc.strip().split(' ')
            for loc_ in loc:
                eid2wid.append([str(entity2id[entity_str]), "0", str(word2id[loc_])])
        if typ != '':
            typ = typ.strip().split(' ')
            for typ_ in typ:
                eid2wid.append([str(entity2id[entity_str]), "1", str(word2id[typ_])])
        eid2wid.append([str(entity2id[entity_str]), "2", str(word2id[w1])])
    else:
        eid2wid.append([str(entity2id[entity_str]), "2", str(word2id[w1])])

with open("e-w-graph.txt", "w") as f:
    for line in eid2wid:
        f.write("\t".join(line)+'\n')




