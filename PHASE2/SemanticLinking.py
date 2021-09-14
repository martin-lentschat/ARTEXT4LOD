# coding: utf-8

import os
import re
import ast
# import json
import spacy
# import stanza
# import math
import functools
import pandas as pd
from tqdm import tqdm
# from nltk.tokenize import sent_tokenize, word_tokenize
# from multiprocessing import Pool
# import matcher


@functools.lru_cache(maxsize=None)
def clean(txt):
    txt = txt.lower()
    txt = re.sub('[^0-9a-z]', '', txt)
    return txt


@functools.lru_cache(maxsize=None)
def vectorize(word, nlp):
    return nlp(word)


@functools.lru_cache(maxsize=None)
def standard(txt):
    return '_'.join([x[0].upper()+x[1:].lower() for x in re.split('[ _]', txt)])


def getEmpty(d):
    return [x for x in d if not d[x][0]]


def getExist(d):
    return [x for x in d if d[x][0]]


def getOriginalValue(ori, typ):
    origin = ast.literal_eval(ori)
    if typ == 'ADDIMENTIONNAL':
        return [' '.join(origin[0][0]), '']
    elif typ == 'SYMBOLIC':
        return [' '.join(origin[0]), '']
    elif typ == 'QUANTITY':
        return [' '.join(origin[0][0]), ' '.join(origin[1][0])]


def getCoref(instances, existarg):
    df = pd.DataFrame('', columns=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence', 'Original_value', 'Node', 'Window', 'Type'])
    for i in existarg:
        df = df.append(instances[(instances.Argument == i) & existarg[i].isin(instances.Original_value)])
    return df


def root_term(liste):
    while type(liste[0]) == list:
        liste = liste[0]
    return liste


def similarity_list(list1, list2):
    res = []
    for i in list1:
        for j in list2:
            if i.vector_norm and j.vector_norm:
                res.append(i.similarity(j))
    return res


def averaging(values):
    h, a, m = {}, {}, {}
    for v in values:
        numbers = [x for x in values[v] if x != 0]
        if numbers:
            h[v] = len(numbers) / (sum([1/x for x in numbers]))  # harmonic mean
            a[v] = sum(numbers) / len(numbers)  # arithmetic mean
            m[v] = max(numbers)
        else:
            h[v] = 0
            a[v] = 0
            m[v] = 0
    return h, a, m


def semanticScores(candidates, instances, tablearg, nlp):
    Scores = {}
    for c_id in list(set(candidates.Clone.values)):
        Scores[c_id] = []
    for a in tablearg:
        args = instances[instances.Argument == a]
        arg_ori = []
        arg_att = []
        for a_id in list(set(args.Clone.values)):
            arg = args[args.Clone == a_id]
            ori_table = [clean(x) for x in tablearg[a][0]]
            atta_table = clean(tablearg[a][1]).split(' ')
            if arg.Type.values[0] == 'SYMBOLIC':
                ori_txt = list(set([clean(''.join(x[0])) for x in [ast.literal_eval(i.Original_value) for i in arg.itertuples()]]))
            else:
                ori_txt = list(set([clean(''.join(x[0][0])) for x in [ast.literal_eval(i.Original_value) for i in arg.itertuples()]]))
            atta_txt = list(set([clean(''.join(x[0])) for x in [ast.literal_eval(i.Attached_value) for i in arg.itertuples()]]))
            if [x for x in ori_table if [y for y in ori_txt if x in y or y in x]]:
                arg_ori = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in arg.Original_value.values]))
                arg_att = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in arg.Attached_value.values]))
                break

            elif [x for x in atta_table if [y for y in atta_txt if x in y or y in x]]:
                arg_ori = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in arg.Original_value.values]))
                arg_att = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in arg.Attached_value.values]))
                break
        if arg_ori:
            arg_ori = [vectorize(x, nlp) for x in arg_ori]
            arg_att = [vectorize(x, nlp) for x in arg_att]
            for c_id in list(set(candidates.Clone.values)):
                candi_val = candidates[candidates.Clone == c_id]
                candi_ori = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in candi_val.Original_value.values]))
                candi_att = list(set([' '.join([x[1:-1] for x in re.findall("'.+?'", str(y))]) for y in candi_val.Attached_value.values]))
                candi_ori = [vectorize(x, nlp) for x in candi_ori]
                candi_att = [vectorize(x, nlp) for x in candi_att]
                Scores[c_id] = Scores[c_id] + similarity_list(arg_ori, candi_ori)
                Scores[c_id] = Scores[c_id] + similarity_list(arg_ori, candi_att)
                Scores[c_id] = Scores[c_id] + similarity_list(arg_att, candi_ori)
                Scores[c_id] = Scores[c_id] + similarity_list(arg_att, candi_att)

    ScoresH, ScoresA, ScoresM = averaging(Scores)
    candidates['Semantic_ScoresH'] = candidates.apply(lambda row: ScoresH[row.Clone] if row.Clone in ScoresH else 0, axis=1)
    candidates['Semantic_ScoresA'] = candidates.apply(lambda row: ScoresH[row.Clone] if row.Clone in ScoresA else 0, axis=1)
    candidates['Semantic_ScoresM'] = candidates.apply(lambda row: ScoresM[row.Clone] if row.Clone in ScoresM else 0, axis=1)
    return candidates


def complete(candidates, instances, tablearg, nlp, top):
    if len(candidates) == 0:
        resH, resA, resM = [['', '']], [['', '']], [['', '']]
    else:
        resH, resA, resM = [], [], []
        candidates = semanticScores(candidates, instances, tablearg, nlp)
        for f in candidates.sample(frac=1).sort_values(by=['Semantic_ScoresH']).head(top).itertuples():
            resH.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
        for f in candidates.sample(frac=1).sort_values(by=['Semantic_ScoresA']).head(top).itertuples():
            resA.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
        for f in candidates.sample(frac=1).sort_values(by=['Semantic_ScoresM']).head(top).itertuples():
            resM.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
    return resH, resA, resM


def getSemanticScore(model, tresh, top):
    nlp = spacy.load(model)
    os.environ["SPACY_WARNING_IGNORE"] = "W008"
    partial_rel = pd.read_csv(r'datatables/resTables.csv')

    # if 'clones.csv' not in os.listdir('workfiles'):
    # detectClone(instances)

    ScoresH = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                          'Caption', 'Segment', 'Document'])
    ScoresA = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                          'Caption', 'Segment', 'Document'])
    ScoresM = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                          'Caption', 'Segment', 'Document'])

    Allinstances = pd.read_csv('workfiles/'+str(tresh)+'clones.csv', encoding='utf-8')

    for n in tqdm(range(0, len(partial_rel))):
        result_argH = ast.literal_eval(partial_rel.loc[n, 'Result_Argument'])
        result_argA = result_argH.copy()
        result_argM = result_argH.copy()
        argumentsH = ast.literal_eval(partial_rel.loc[n, 'Arguments'])
        argumentsA = argumentsH.copy()
        argumentsM = argumentsH.copy()

        missing_res_arg = getEmpty(result_argH)
        missing_second_arg = getEmpty(argumentsH)
        tablearg = {**result_argH, **argumentsH}

        instances = Allinstances[Allinstances.Document == partial_rel.loc[n, 'Document']+'.xml']

        for k in tablearg.copy():
            if k in missing_res_arg or k in missing_second_arg:
                del tablearg[k]

        for miss in missing_res_arg:
            candidates = instances[instances.Argument == miss].copy()
            result_argH[miss], result_argA[miss], result_argM[miss] = complete(candidates, instances, tablearg, nlp, top)
        for miss in missing_second_arg:
            candidates = instances[instances.Argument == miss].copy()
            argumentsH[miss], argumentsA[miss], argumentsM[miss] = complete(candidates, instances, tablearg, nlp, top)

        ScoresH = ScoresH.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                                    'Result_Argument': [result_argH],
                                                    'Arguments': [argumentsH],
                                                    'Table': [partial_rel.loc[n, 'Table']],
                                                    'Caption': [partial_rel.loc[n, 'Caption']],
                                                    'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                                    'Document': [partial_rel.loc[n, 'Document']]}),
                                 ignore_index=True)
        ScoresA = ScoresA.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                                    'Result_Argument': [result_argA],
                                                    'Arguments': [argumentsA],
                                                    'Table': [partial_rel.loc[n, 'Table']],
                                                    'Caption': [partial_rel.loc[n, 'Caption']],
                                                    'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                                    'Document': [partial_rel.loc[n, 'Document']]}),
                                 ignore_index=True)
        ScoresM = ScoresM.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                                    'Result_Argument': [result_argM],
                                                    'Arguments': [argumentsM],
                                                    'Table': [partial_rel.loc[n, 'Table']],
                                                    'Caption': [partial_rel.loc[n, 'Caption']],
                                                    'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                                    'Document': [partial_rel.loc[n, 'Document']]}),
                                 ignore_index=True)
    return ScoresH, ScoresA, ScoresM


if __name__ == '__main__':
    print('Semantic Linking')
