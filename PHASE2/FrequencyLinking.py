# coding: utf-8

# import os
import re
import ast
# import json
# import spacy
# import stanza
import math
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
def standard(txt):
    split_txt = re.split('[ _]', txt)
    return '_'.join([x[0].upper()+x[1:].lower() for x in split_txt])


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

#
# def detectClone(instances):
#     clone = pd.DataFrame(None, columns=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence', 'Original_value', 'Node', 'Window', 'Type', 'Clone'], dtype=object)
#     n = 0
#     arg = instances.Argument.values
#     doc = instances.Document.values
#     catego = instances.Type.values
#     node = instances.Node.values
#     ori = instances.Original_value.values
#
#     for i in tqdm(range(0, len(instances))):
#         n += 1
#
#         if catego[i] == 'SYMBOLIC':
#             c = instances.loc[(instances.Argument == arg[i]) &
#                               (instances.Document == doc[i])]
#             for j in c.itertuples():
#                 if ast.literal_eval(j.Node)[0] == ast.literal_eval(node[i])[0]:
#                     clone = clone.append({'Argument': j.Argument,
#                                           'Attached_value': j.Attached_value,
#                                           'Document': j.Document,
#                                           'Segment': j.Segment,
#                                           'Sentence': j.Sentence,
#                                           'Original_value': j.Original_value,
#                                           'Node': j.Node,
#                                           'Window': j.Window,
#                                           'Type': j.Type,
#                                           'Clone': n}, ignore_index=True)
#
#         else:
#             c = instances.loc[(instances.Argument == arg[i]) &
#                               (instances.Document == doc[i])]
#             for j in c.itertuples():
#                 if ast.literal_eval(j.Original_value)[0][0][0] == ast.literal_eval(ori[i])[0][0][0]:
#                     clone = clone.append({'Argument': j.Argument,
#                                           'Attached_value': j.Attached_value,
#                                           'Document': j.Document,
#                                           'Segment': j.Segment,
#                                           'Sentence': j.Sentence,
#                                           'Original_value': j.Original_value,
#                                           'Node': j.Node,
#                                           'Window': j.Window,
#                                           'Type': j.Type,
#                                           'Clone': n}, ignore_index=True)
#
#     clone.drop_duplicates(subset=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence',
#                                   'Original_value', 'Node', 'Window', 'Type'], keep='first')
#     clone.to_csv('workfiles/clones.csv', encoding='utf-8')
#     # return clone


def getCoref(instances, existarg):
    df = pd.DataFrame('', columns=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence', 'Original_value', 'Node', 'Window', 'Type'])
    for i in existarg:
        df = df.append(instances[(instances.Argument == i) & existarg[i].isin(instances.Original_value)])
    return df


def root_term(liste):
    while type(liste[0]) == list:
        liste = liste[0]
    return liste


def coocuScores(candidates, instances, tablearg, context_weight, term_weight):
    PMIs, DICEs, JACCARDs = {}, {}, {}
    for c_id in list(set(candidates.Clone.values)):
        PMIs[c_id] = 0
        DICEs[c_id] = 0
        JACCARDs[c_id] = 0
    arg_val = pd.DataFrame(None, columns=['Sentence', 'Window', 'Segment', 'Document', 'Weight'])
    for a in tablearg:
        args = instances[instances.Argument == a]
        for a_id in list(set(args.Clone.values)):
            arg = args[args.Clone == a_id]
            ori_table = [clean(x) for x in tablearg[a][0]]
            atta_table = clean(tablearg[a][1]).split(' ')
            if arg.Type.values[0] == 'SYMBOLIC':
                ori_txt = list(set([clean(''.join(x[0])) for x in [ast.literal_eval(i.Original_value) for i in arg.itertuples()]]))
            else:
                ori_txt = list(set([clean(''.join(x[0][0])) for x in [ast.literal_eval(i.Original_value) for i in arg.itertuples()]]))
            atta_txt = list(set([clean(''.join(x[0])) for x in [ast.literal_eval(i.Attached_value) for i in arg.itertuples()]]))

            # if clean(tablearg[a][0]) in [clean(''.join(root_term(ast.literal_eval(i.Original_value)))) for i in arg.itertuples()]\
            #         and term_weight['Original_value']:
            if [x for x in ori_table if [y for y in ori_txt if x in y or y in x]] and term_weight['Original_value']:
                arg_val = arg_val.append(pd.DataFrame({'Sentence': arg.Sentence.values,
                                                       'Window': arg.Window.values,
                                                       'Segment': arg.Segment.values,
                                                       'Document': arg.Document.values,
                                                       'Weight': term_weight['Original_value']}),
                                         ignore_index=True)
                break

            # elif clean(tablearg[a][1]) in [clean(''.join(ast.literal_eval(i.Attached_value)[0])) for i in arg.itertuples()]\
            #         and term_weight['Attached_value']:
            elif [x for x in atta_table if [y for y in atta_txt if x in y or y in x]] and term_weight['Attached_value']:
                arg_val = arg_val.append(pd.DataFrame({'Sentence': arg.Sentence.values,
                                                       'Window': arg.Window.values,
                                                       'Segment': arg.Segment.values,
                                                       'Document': arg.Document.values,
                                                       'Weight': term_weight['Attached_value']}),
                                         ignore_index=True)
                break

            elif term_weight['Argument']:
                arg_val = arg_val.append(pd.DataFrame({'Sentence': arg.Sentence.values,
                                                       'Window': arg.Window.values,
                                                       'Segment': arg.Segment.values,
                                                       'Document': arg.Document.values,
                                                       'Weight': term_weight['Argument']}),
                                         ignore_index=True)
                break


        if len(arg_val):
            for c_id in list(set(candidates.Clone.values)):
                s = {'pmi': 0, 'dice': 0, 'jaccard': 0}
                candi_val = candidates[candidates.Clone == c_id]
                cooc = 0
                candi_con = []
                arg_con = []
                for wc in context_weight:
                    context = list(set(candi_val[wc].values))
                    candi_con += context
                    for n in range(0, len(arg_val)):
                        arg_con += [arg_val.loc[n, wc]]
                        if arg_val.loc[n, wc] in context:
                            cooc += context_weight[wc] * arg_val.loc[n, 'Weight']
                if cooc != 0:
                    s['pmi'] += math.log2((cooc/len(instances)) / ((len(candi_val)/len(instances)) * (len(arg_val)/len(instances))))
                    s['dice'] += 2 * cooc / len(list(set(candi_con))) + len(list(set(arg_con)))
                    s['jaccard'] += cooc / len(list(set(candi_con + arg_con)))
                PMIs[c_id] += s['pmi']
                DICEs[c_id] += s['dice']
                JACCARDs[c_id] += s['jaccard']

    candidates['PMI_FreqScores'] = candidates.apply(lambda row: PMIs[row.Clone] if row.Clone in PMIs else 0, axis=1)
    candidates['DICE_FreqScores'] = candidates.apply(lambda row: DICEs[row.Clone] if row.Clone in DICEs else 0, axis=1)
    candidates['JACCARD_FreqScores'] = candidates.apply(lambda row: JACCARDs[row.Clone] if row.Clone in JACCARDs else 0, axis=1)
    return candidates


def complete(candidates, instances, tablearg, context_weight, term_weight, top):
    if len(candidates) == 0:
        res = [[['', '']], [['', '']], [['', '']]]
    else:
        res_pmi = []
        res_dice = []
        res_jaccard = []
        candidates = coocuScores(candidates, instances, tablearg, context_weight, term_weight)
        for f in candidates.sample(frac=1).sort_values(by=['PMI_FreqScores']).head(top).itertuples():
            res_pmi.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
        for f in candidates.sample(frac=1).sort_values(by=['DICE_FreqScores']).head(top).itertuples():
            res_dice.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])
        for f in candidates.sample(frac=1).sort_values(by=['JACCARD_FreqScores']).head(top).itertuples():
            res_jaccard.append([getOriginalValue(f.Original_value, f.Type), ' '.join(ast.literal_eval(f.Attached_value)[0])])

        # PMI = candidates.sample(frac=1).sort_values(by=['PMI_FreqScores']).head(1)
        # DICE = candidates.sample(frac=1).sort_values(by=['DICE_FreqScores']).head(1)
        # JACCARD = candidates.sample(frac=1).sort_values(by=['JACCARD_FreqScores']).head(1)
        # res = [[getOriginalValue(PMI), ' '.join([ast.literal_eval(x)[0] for x in PMI.Attached_value][0])],
        #        [getOriginalValue(DICE), ' '.join([ast.literal_eval(x)[0] for x in DICE.Attached_value][0])],
        #        [getOriginalValue(JACCARD), ' '.join([ast.literal_eval(x)[0] for x in JACCARD.Attached_value][0])]]
        res = [res_pmi, res_dice, res_jaccard]
    return res


# def complet_res_arg(miss, instances, tablearg, context_weight, term_weight):
#     candidates = instances[instances.Argument == miss].copy()
#     return miss, complete(candidates, instances, tablearg, context_weight, term_weight)


# def complet_second_arg(miss, instances, tablearg, context_weight, term_weight):
#     candidates = instances[instances.Argument == miss].copy()
#     return miss, complete(candidates, instances, tablearg, context_weight, term_weight)


def getFrequencyScore(context_weight, term_weight, tresh, top):
    partial_rel = pd.read_csv(r'datatables/resTables.csv')

    # if 'clones.csv' not in os.listdir('workfiles'):
    # detectClone(instances)

    PMI = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                      'Caption', 'Segment', 'Document'])
    DICE = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                       'Caption', 'Segment', 'Document'])
    JACCARD = pd.DataFrame(None, columns=['Relation', 'Result_Argument', 'Arguments', 'Table',
                                          'Caption', 'Segment', 'Document'])

    Allinstances = pd.read_csv('workfiles/'+str(tresh)+'clones.csv', encoding='utf-8')

    for n in tqdm(range(0, len(partial_rel))):
        result_arg = ast.literal_eval(partial_rel.loc[n, 'Result_Argument'])
        result_argPMI, result_argDICE, result_argJACCARD = result_arg.copy(), result_arg.copy(), result_arg.copy()
        arguments = ast.literal_eval(partial_rel.loc[n, 'Arguments'])
        argumentsPMI, argumentsDICE, argumentsJACCARD = arguments.copy(), arguments.copy(), arguments.copy()

        missing_res_arg = getEmpty(result_arg)
        missing_second_arg = getEmpty(arguments)
        tablearg = {**result_arg, **arguments}

        for k in tablearg.copy():
            if k in missing_res_arg or k in missing_second_arg:
                del tablearg[k]

        instances = Allinstances[Allinstances.Document == partial_rel.loc[n, 'Document']+'.xml']

        for miss in missing_res_arg:
            candidates = instances[instances.Argument == miss].copy()
            res = complete(candidates, instances, tablearg, context_weight, term_weight, top)
            result_argPMI[miss] = res[0]
            result_argDICE[miss] = res[1]
            result_argJACCARD[miss] = res[2]
        for miss in missing_second_arg:
            candidates = instances[instances.Argument == miss].copy()
            res = complete(candidates, instances, tablearg, context_weight, term_weight, top)
            argumentsPMI[miss] = res[0]
            argumentsDICE[miss] = res[1]
            argumentsJACCARD[miss] = res[2]

        PMI = PMI.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                            'Result_Argument': [result_argPMI],
                                            'Arguments': [argumentsPMI],
                                            'Table': [partial_rel.loc[n, 'Table']],
                                            'Caption': [partial_rel.loc[n, 'Caption']],
                                            'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                            'Document': [partial_rel.loc[n, 'Document']]}),
                         ignore_index=True)
        DICE = DICE.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                              'Result_Argument': [result_argDICE],
                                              'Arguments': [argumentsDICE],
                                              'Table': [partial_rel.loc[n, 'Table']],
                                              'Caption': [partial_rel.loc[n, 'Caption']],
                                              'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                              'Document': [partial_rel.loc[n, 'Document']]}),
                           ignore_index=True)
        JACCARD = JACCARD.append(pd.DataFrame(data={'Relation': [standard(partial_rel.loc[n, 'Relation'])],
                                                    'Result_Argument': [result_argJACCARD],
                                                    'Arguments': [argumentsJACCARD],
                                                    'Table': [partial_rel.loc[n, 'Table']],
                                                    'Caption': [partial_rel.loc[n, 'Caption']],
                                                    'Segment': [ast.literal_eval(partial_rel.loc[n, 'Segment'])],
                                                    'Document': [partial_rel.loc[n, 'Document']]}),
                                 ignore_index=True)
    # PMI.to_csv('workfiles\PMI.csv', encoding='utf-8')
    # DICE.to_csv('workfiles\DICE.csv', encoding='utf-8')
    # JACCARD.to_csv('workfiles\JACCARD.csv', encoding='utf-8')
    return PMI, DICE, JACCARD


if __name__ == '__main__':
    print('Structural Linking')
