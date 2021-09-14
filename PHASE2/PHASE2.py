# coding: utf-8

import os
import gc
# import re
import ast
# import json
# import spacy
# import stanza
# import Levenshtein
import re

import pandas as pd
from tqdm import tqdm
# from nltk.tokenize import sent_tokenize, word_tokenize

import TablesParser
import StructuralLinking
import FrequencyLinking
import SemanticLinking
import eval

# import cProfile
# import pstats


def detectClone(instances, tresh):
    clone = pd.DataFrame(None, columns=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence', 'Original_value', 'Node', 'Window', 'Type', 'Clone'], dtype=object)
    n = 0
    arg = instances.Argument.values
    doc = instances.Document.values
    catego = instances.Type.values
    node = instances.Node.values
    ori = instances.Original_value.values

    for i in tqdm(range(0, len(instances))):
        n += 1

        if catego[i] == 'SYMBOLIC':
            c = instances.loc[(instances.Argument == arg[i]) &
                              (instances.Document == doc[i])]
            for j in c.itertuples():
                if ast.literal_eval(j.Node)[0] == ast.literal_eval(node[i])[0]:
                    clone = clone.append({'Argument': j.Argument,
                                          'Attached_value': j.Attached_value,
                                          'Document': j.Document,
                                          'Segment': j.Segment,
                                          'Sentence': j.Sentence,
                                          'Original_value': j.Original_value,
                                          'Node': j.Node,
                                          'Window': j.Window,
                                          'Type': j.Type,
                                          'Clone': n}, ignore_index=True)

        else:
            c = instances.loc[(instances.Argument == arg[i]) &
                              (instances.Document == doc[i])]
            for j in c.itertuples():
                if ast.literal_eval(j.Original_value)[0][0][0] == ast.literal_eval(ori[i])[0][0][0]:
                    clone = clone.append({'Argument': j.Argument,
                                          'Attached_value': j.Attached_value,
                                          'Document': j.Document,
                                          'Segment': j.Segment,
                                          'Sentence': j.Sentence,
                                          'Original_value': j.Original_value,
                                          'Node': j.Node,
                                          'Window': j.Window,
                                          'Type': j.Type,
                                          'Clone': n}, ignore_index=True)

    clone.drop_duplicates(subset=['Argument', 'Attached_value', 'Document', 'Segment', 'Sentence',
                                  'Original_value', 'Node', 'Window', 'Type'], keep='first')
    clone.to_csv('workfiles/'+str(tresh)+'clones.csv', encoding='utf-8')
    # return clone


def filtre(instances, docs, relevance):
    instances['Argument'] = instances['Argument'].apply(lambda x: StructuralLinking.standard(x))
    docs = list(set([x+'.xml' for x in docs]))
    instances = instances[instances.Document.isin(docs)]
    res = pd.DataFrame(None)

    prefs_scores = {'SYMBOLIC': ['DC_Tree', 'TF_classic_term'],
                    'ADDIMENTIONNAL': 'ICF_segment_term_top',
                    'QUANTITY': 'ICF_segment_term_top',
                    'packaging': ['DC_Tree', 'TF_classic_term'],
                    'method': ['DC_Tree', 'TF_classic_term'],
                    'impact_factor_component': 'ICF_segment_term_top',
                    'partial_pressure': 'ICF_segment_term_top',
                    'relative_humidity': 'ICF_segment_term_top',
                    'component_qty_value': 'ICF_segment_term_top',
                    'temperature': 'ICF_segment_term_top',
                    'thickness': 'ICF_segment_term_top',
                    'permeability': 'ICF_segment_term_top',
                    'h2o_permeability': 'ICF_segment_term_top',
                    'o2_permeability': 'ICF_segment_term_top',
                    'co2_permeability': 'ICF_segment_term_top',
                    'partial_pressure_difference': 'ICF_segment_term_top'}

    if 'VALID' in instances.columns:
        for a in list(set(instances.Argument.values)):
            temp = instances[instances.Argument == a]
            # res = res.append(temp.sort_values(by='VALID', ascending=False).head(int(len(temp)*relevance[a.lower()][0])), ignore_index=True)
            res = res.append(temp.sort_values(by='VALID', ascending=False).head(int(len(temp)*relevance)), ignore_index=True)
    else:
        for a in list(set(instances.Argument.values)):
            temp = instances[instances.Argument == a]
            # if type(relevance[a.lower()][1]) == list:
            if type(prefs_scores[a.lower()]) == list:
                # for s in relevance[a.lower()][1]:
                for s in prefs_scores[a.lower()]:
                    # temp = temp.sort_values(by=s, ascending=False).head(int(len(temp)*relevance[a.lower()][0]))
                    temp = temp.sort_values(by=s, ascending=False).head(int(len(temp)*relevance))
                res = res.append(temp, ignore_index=True)
            else:
                # res = res.append(temp.sort_values(by=relevance[a.lower()][1], ascending=False).head(int(len(temp)*relevance[a.lower()][0])), ignore_index=True)
                res = res.append(temp.sort_values(by=prefs_scores[a.lower()], ascending=False).head(int(len(temp)*relevance)), ignore_index=True)
    return res


# def createparams():
#     params = []
#     files = ['valid_golden-all', 'resFINAL']
#     treshold = [1, .8, .6, .4]
#     context_weight = [{'Sentence': 1, 'Window': 1, 'Segment': .1, 'Document': .1},  #close
#                       {'Sentence': 1, 'Window': 1, 'Segment': 1, 'Document': .1},  #section
#                       {'Sentence': 1, 'Window': 1, 'Segment': 1, 'Document': 1}]  #gene
#     term_weight = [{'Original_value': 1, 'Attached_value': .1, 'Argument': .1},  #pure
#                    {'Original_value': 1, 'Attached_value': 1, 'Argument': .1},  #concept
#                    {'Original_value': 1, 'Attached_value': 1, 'Argument': 1}] #gene
#
#     for f in files:
#         param = {'file': f+'.csv'}
#         if f == 'valid_golden-all':
#             param['treshold'] = 1
#             for c in context_weight:
#                 for t in term_weight:
#                     param['context_weight'] = c
#                     param['term_weight'] = t
#                     if param not in params:
#                         params.append(param.copy())
#         else:
#             for tresh in treshold:
#                 param['treshold'] = tresh
#                 for c in context_weight:
#                     for t in term_weight:
#                         param['context_weight'] = c
#                         param['term_weight'] = t
#                         params.append(param.copy())
    # relevance = {'SYMBOLIC': [.7, ['DC_Tree', 'TF_classic_term']],
    #              'ADDIMENTIONNAL': [.7, 'ICF_segment_term_top'],
    #              'QUANTITY': [.7, 'ICF_segment_term_top'],
    #              'packaging': [.7, ['DC_Tree', 'TF_classic_term']],
    #              'method': [.7, ['DC_Tree', 'TF_classic_term']],
    #              'impact_factor_component': [.7, 'ICF_segment_term_top'],
    #              'partial_pressure': [.7, 'ICF_segment_term_top'],
    #              'relative_humidity': [.7, 'ICF_segment_term_top'],
    #              'component_qty_value': [.7, 'ICF_segment_term_top'],
    #              'temperature': [.7, 'ICF_segment_term_top'],
    #              'thickness': [.7, 'ICF_segment_term_top'],
    #              'permeability': [.7, 'ICF_segment_term_top'],
    #              'h2o_permeability': [.7, 'ICF_segment_term_top'],
    #              'o2_permeability': [.7, 'ICF_segment_term_top'],
    #              'co2_permeability': [.7, 'ICF_segment_term_top'],
    #              'partial_pressure_difference': [.7, 'ICF_segment_term_top']}
    # DC_Tree
    # ICF_segment_arg_bot	ICF_segment_arg_top
    # ICF_segment_term_bot	ICF_segment_term_top
    # IDF_classic_term
    # TF_classic_term
    # TF_segment_arg_bot	TF_segment_arg_top
    # TF_segment_term_bot	TF_segment_term_top
    # return params


# def build_name(file, treshold, context_weight, term_weight):
#     name = re.split('\.', file)[0]
#     name += str(treshold)
#     name += ''.join([x+str(context_weight[x]) for x in context_weight])
#     name += ''.join([x+str(term_weight[x]) for x in term_weight])
#     return name


def iteration():

    top = [('top 1', 1), ('top 3', 3), ('top 5', 5), ('top 10', 10)]
    files = ['valid_golden-all', 'resFINAL']
    treshold = [1, .8, .6, .4]
    context_weight = [{'Sentence': 1, 'Window': 0, 'Segment': 0, 'Document': 0},  # que sentence
                      {'Sentence': 0, 'Window': 1, 'Segment': 0, 'Document': 0},  # que window
                      {'Sentence': 0, 'Window': 0, 'Segment': 1, 'Document': 0},  # que segment
                      {'Sentence': 0, 'Window': 0, 'Segment': 0, 'Document': 1},  # que document
                      {'Sentence': 1, 'Window': 1, 'Segment': 0, 'Document': 0},  # close
                      {'Sentence': 1, 'Window': 1, 'Segment': 1, 'Document': 0},  # section
                      {'Sentence': 1, 'Window': 1, 'Segment': 1, 'Document': 1}]  # generic
    term_weight = [{'Original_value': 1, 'Attached_value': 0, 'Argument': 0},  # pure
                   {'Original_value': 0, 'Attached_value': 1, 'Argument': 0},  # attachment
                   {'Original_value': 0, 'Attached_value': 0, 'Argument': 1},  # concept
                   {'Original_value': 1, 'Attached_value': 1, 'Argument': 0},  # lexical
                   {'Original_value': 1, 'Attached_value': 1, 'Argument': 1}]  # generic

    models = ['en_core_sci_lg',         # large base scispacy
              'en_core_sci_scibert',    # BioBert in scispacy format
              'en_ner_craft_md',        # scispacy on CRAFT corpus
              'en_ner_jnlpba_md',       # scispacy on JNLPBA corpus
              'en_ner_bc5cdr_md',       # scispacy on BC5CDR corpus
              'en_ner_bionlp13cg_md',   # scispacy on BIONLP13CG corpus
              'en_core_web_lg',         # large base spacy
              'en_core_web_trf'         # Bert in spacy format
              ]
    # equation = ['harmonic', 'max']

    for t in top:
        for file in files:
            args = pd.read_csv('input_files/'+file+'.csv', encoding='utf-8')
            for tresh in treshold:
                path = t[0] + re.split('\.', file)[0] + str(tresh)
                if file == 'valid_golden-all' and tresh != 1:
                    continue
                if path not in os.listdir('results'):
                    os.mkdir('results/'+path)
                    os.mkdir('results/'+path+'/structural')
                    os.mkdir('results/'+path+'/frequency')
                    os.mkdir('results/'+path+'/semantic')

                print('#'*15)
                print('# PREPARATION')
                with open('workfiles/candidats.csv', 'w', encoding='utf-8') as f:
                    instances = filtre(args, pd.read_csv('datatables/resTables.csv', encoding='utf-8').Document.values, tresh)
                    instances.to_csv(f, encoding='utf-8')
                if str(tresh)+'clones.csv' not in os.listdir('workfiles'):
                    detectClone(instances, tresh)
                print('# OK!')

                print('#'*15)
                print('# STRUCTURAL SIMPLE')
                print('file: '+file)
                print('treshold: '+str(tresh))
                print('selection: '+t[0])
                if 'structuralSimple.csv' not in os.listdir('results/'+path+'/structural'):
                    structuralSimple = StructuralLinking.getStructuralScore(instances, False, t[1])
                    structuralSimple.to_csv('results/'+path+'/structural/structuralSimple.csv', encoding='utf-8')
                structuralSimple = pd.read_csv('results/'+path+'/structural/structuralSimple.csv', encoding='utf-8')
                eval.evaluation(structuralSimple, 'results/'+path+'/structural/EVAL_structuralSimple')
                del structuralSimple

                print('#'*15)
                print('# STRUCTURAL GUIDED')
                print('file: '+file)
                print('treshold: '+str(tresh))
                print('selection: '+t[0])
                if 'structuralGuided.csv' not in os.listdir('results/'+path+'/structural'):
                    structuralGuided = StructuralLinking.getStructuralScore(instances, True, t[1])
                    structuralGuided.to_csv('results/'+path+'/structural/structuralGuided.csv', encoding='utf-8')
                structuralGuided = pd.read_csv('results/'+path+'/structural/structuralGuided.csv', encoding='utf-8')
                eval.evaluation(structuralGuided, 'results/'+path+'/structural/EVAL_structuralGuided')
                del structuralGuided

                for con_wei in context_weight:
                    for ter_wei in term_weight:
                        name = ''.join([x+str(con_wei[x]) for x in con_wei])
                        name += ''.join([x+str(ter_wei[x]) for x in ter_wei])
                        if name not in os.listdir('results/'+path+'/frequency'):
                            os.mkdir('results/'+path+'/frequency/'+name)

                        print('#'*15)
                        print('# FREQUENCY')
                        print('file: '+file)
                        print('treshold: '+str(tresh))
                        print('context: '+str(con_wei))
                        print('term: '+str(ter_wei))
                        print('selection: '+t[0])
                        if 'frequencyPMI.csv' not in os.listdir('results/'+path+'/frequency/'+name):
                            frequencyPMI, frequencyDICE, frequencyJACCARD = FrequencyLinking.getFrequencyScore(con_wei, ter_wei, tresh, t[1])
                            frequencyPMI.to_csv('results/'+path+'/frequency/'+name+'/frequencyPMI.csv', encoding='utf-8')
                            frequencyDICE.to_csv('results/'+path+'/frequency/'+name+'/frequencyDICE.csv', encoding='utf-8')
                            frequencyJACCARD.to_csv('results/'+path+'/frequency/'+name+'/frequencyJACCARD.csv', encoding='utf-8')
                        frequencyPMI = pd.read_csv('results/'+path+'/frequency/'+name+'/frequencyPMI.csv', encoding='utf-8')
                        eval.evaluation(frequencyPMI, 'results/'+path+'/frequency/'+name+'/EVAL_frequencyPMI')
                        frequencyDICE = pd.read_csv('results/'+path+'/frequency/'+name+'/frequencyDICE.csv', encoding='utf-8')
                        eval.evaluation(frequencyDICE, 'results/'+path+'/frequency/'+name+'/EVAL_frequencyDICE')
                        frequencyJACCARD = pd.read_csv('results/'+path+'/frequency/'+name+'/frequencyJACCARD.csv', encoding='utf-8')
                        eval.evaluation(frequencyJACCARD, 'results/'+path+'/frequency/'+name+'/EVAL_frequencyJACCARD')
                        del frequencyPMI, frequencyDICE, frequencyJACCARD

                for model in models:
                    print('#'*15)
                    print('# SEMANTIC')
                    print('file: '+file)
                    print('treshold: '+str(tresh))
                    print('model: '+model)
                    print('selection: '+t[0])
                    if 'harmonic'+model+'.csv' not in os.listdir('results/'+path+'/semantic'):
                        semanticH, semanticA, semanticM = SemanticLinking.getSemanticScore(model, tresh, t[1])
                        semanticH.to_csv('results/'+path+'/semantic/harmonic'+model+'.csv', encoding='utf-8')
                        semanticA.to_csv('results/'+path+'/semantic/arithmetic'+model+'.csv', encoding='utf-8')
                        semanticM.to_csv('results/'+path+'/semantic/max'+model+'.csv', encoding='utf-8')
                    semanticH = pd.read_csv('results/'+path+'/semantic/harmonic'+model+'.csv', encoding='utf-8')
                    semanticA = pd.read_csv('results/'+path+'/semantic/arithmetic'+model+'.csv', encoding='utf-8')
                    semanticM = pd.read_csv('results/'+path+'/semantic/max'+model+'.csv', encoding='utf-8')
                    eval.evaluation(semanticH, 'results/'+path+'/semantic/EVAL_harmonic'+model)
                    eval.evaluation(semanticA, 'results/'+path+'/semantic/EVAL_arithmetic'+model)
                    eval.evaluation(semanticM, 'results/'+path+'/semantic/EVAL_max'+model)
                    del semanticH, semanticA, semanticM
                    
                gc.collect()


def main():
    if 'resTables.csv' not in os.listdir('datatables'):
        with open('datatables/resTables.csv', 'w', encoding='utf-8') as f:
            partial_rel = TablesParser.parse_tables()
            partial_rel.to_csv(f, encoding='utf-8')
    iteration()
    eval.analyse_results()


if __name__ == '__main__':
    main()
