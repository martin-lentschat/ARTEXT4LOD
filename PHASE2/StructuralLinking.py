# coding: utf-8

# import os
import re
import ast
# import json
import spacy
# import stanza
import functools
import pandas as pd
from tqdm import tqdm
# from nltk.tokenize import sent_tokenize, word_tokenize
# from multiprocessing import Pool

import matcher


@functools.lru_cache(maxsize=None)
def standard(txt):
    split_txt = re.split('[ _]', txt)
    return '_'.join([x[0].upper()+x[1:].lower() for x in split_txt])


def getEmpty(d):
    return [x for x in d if not d[x][0]]


def pureCaption(cap):
    while re.findall('table|\d|\.', cap[0]):
        cap = cap[1:]
    return cap


def findProxiScore(candidates, segment, caption, txt, guided, tokenicer):
    text = ' '.join(txt.Text.to_list())

    text_tokens = [x.text.lower() for x in tokenicer(text)]
    text_caption = pureCaption([x.text.lower() for x in tokenicer(caption)])
    candidates['Segment'] = candidates['Segment'].apply(ast.literal_eval)

    reduced = pd.DataFrame(columns=candidates.columns)
    if guided:
        expert_guide = pd.read_csv('input_files/expertguide.csv', encoding='utf-8', sep=';')
        expert_guide['Argument'] = [standard(x) for x in expert_guide.Argument.values]
        expert_guide['Section'] = expert_guide['Section'].apply(ast.literal_eval)
        expert_guide = expert_guide[expert_guide.Argument == candidates.Argument.values[0]].sort_values(by=['Score'], ascending=False)
        for e in expert_guide.itertuples():
            if e.Score == 0:
                break
            for c in candidates.itertuples():
                if list(set(c.Segment) & set(e.Section)):
                    reduced = reduced.append(pd.DataFrame([c[1:]], columns=candidates.columns))
            if len(reduced) > 0:
                candidates = reduced
                reduced = pd.DataFrame(columns=candidates.columns)
                break

    for n in reversed(range(len(segment), 0)):
        for c in candidates.itertuples():
            if len(list(set(segment) & set(c.Segment))) == n:
                reduced = reduced.append(pd.DataFrame([c[1:]], columns=candidates.columns))
        if len(reduced) > 0:
            candidates = reduced
            reduced = pd.DataFrame(columns=candidates.columns)
            break

    sections_order = list(dict.fromkeys([x[0] for x in candidates.Segment.to_list()]))
    if segment[0] in sections_order:
        place = sections_order.index(segment[0])
        dist_end = len(sections_order) - place
        for n in range(1, max(place, dist_end)):
            for c in candidates.itertuples():
                if c.Segment[0] == sections_order[max(place - n, 0)] or c.Segment[0] == sections_order[min(place + n, len(sections_order)-1)]:
                    reduced = reduced.append(pd.DataFrame([c[1:]], columns=candidates.columns))
            if len(reduced) > 0:
                candidates = reduced
                break

    dist_token = 0
    for c in candidates.itertuples():
        if type(ast.literal_eval(c.Original_value)[-1]) != list:
            text_instance = [x.lower() for x in ast.literal_eval(c.Original_value)[0]]
        else:
            text_instance = [x.lower() for x in ast.literal_eval(c.Original_value)[0][0]]
        text_sentence = [x.lower() for x in ast.literal_eval(c.Sentence)]
        posi_table = matcher.match_sequence(text_caption, text_tokens)
        posi_table = posi_table[0][1] if posi_table else 0
        posi_sent = matcher.match_sequence(text_sentence, text_tokens)
        posi_sent = posi_sent[0][1] if posi_sent else 0
        posi_inst = matcher.match_sequence(text_instance, text_sentence)
        posi_inst = posi_inst[0][1] if posi_inst else 0
        """
        stabiliser la reconaissance de tokens pour éviter les différences de découpage entre table, sent, text ...
        -> virer les exeptions
        """
        dist_token = ((abs((posi_sent + posi_inst) - posi_table)) / len(text_tokens)) / 100
    candidates['ProxiScore'] = dist_token

    return candidates


def getOriginalValue(ori, typ):
    origin = ast.literal_eval(ori)
    if typ == 'ADDIMENTIONNAL':
        return [' '.join(origin[0][0]), '']
    elif typ == 'SYMBOLIC':
        return [' '.join(origin[0]), '']
    elif typ == 'QUANTITY':
        return [' '.join(origin[0][0]), ' '.join(origin[1][0])]


def complete(candidates, segment, caption, txt, guided, tokenicer, top):
    if len(candidates) == 0:
        res = [['', '']]
    else:
        res = []
        candidates = findProxiScore(candidates, segment, caption, txt, guided, tokenicer)
        for f in candidates.sort_values(by=['ProxiScore']).head(top).itertuples():
            res.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
    return res


def getStructuralScore(instances, guided, top):
    partial_rel = pd.read_csv(r'datatables/resTables.csv')
    relation_type = [standard(x) for x in partial_rel.Relation.to_list()]
    result_arg = [ast.literal_eval(x) for x in partial_rel.Result_Argument.to_list()]
    arguments = [ast.literal_eval(x) for x in partial_rel.Arguments.to_list()]
    table_name = [x for x in partial_rel.Table.to_list()]
    caption = [x for x in partial_rel.Caption.to_list()]
    segment = [ast.literal_eval(x) for x in partial_rel.Segment.to_list()]
    document = [x for x in partial_rel.Document.to_list()]
    corpus = pd.read_csv(r'input_files\corp.csv', encoding='utf-8')

    # tokenicer = spacy.load("en_core_web_sm")
    tokenicer = spacy.load("en_core_web_lg")

    for n in tqdm(range(0, len(partial_rel))):
        txt = corpus[corpus.Document == document[n]+'.xml']
        missing_res_arg = getEmpty(result_arg[n])
        missing_second_arg = getEmpty(arguments[n])
        for miss in missing_res_arg:
            candidates = instances[(instances.Argument == miss) & (instances.Document == document[n]+'.xml')].copy()
            result_arg[n][miss] = complete(candidates, segment[n], caption[n], txt, guided, tokenicer, top)
        for miss in missing_second_arg:
            candidates = instances[(instances.Argument == miss) & (instances.Document == document[n]+'.xml')].copy()
            arguments[n][miss] = complete(candidates, segment[n], caption[n], txt, guided, tokenicer, top)

    return pd.DataFrame(data={'Relation': relation_type,
                              'Result_Argument': result_arg,
                              'Arguments': arguments,
                              'Table': table_name,
                              'Caption': caption,
                              'Segment': segment,
                              'Document': document})


if __name__ == '__main__':
    print('Structural Linking')
