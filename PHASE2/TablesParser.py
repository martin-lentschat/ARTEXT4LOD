# coding: utf-8

import os
import re
# import ast
# import json
# import spacy
# import stanza
import pandas as pd
# from tqdm import tqdm
from bs4 import BeautifulSoup
# from nltk.tokenize import sent_tokenize, word_tokenize


def clean(txt):
    txt = re.sub('[\n\t ]+', ' ', txt)
    while txt[0] == ' ':
        txt = txt[1:]
    while txt[-1] == ' ':
        txt = txt[:-1]
    return txt


def findValue(col_num, cels, head):
    if col_num == '':
        return ''
    else:
        cel = cels[col_num]
        col = head[col_num]
        # if cel['type'] != col['type']:
        #     print('ERROR\n', col_num, '\n', cels, '\n', head)
        res = [clean(' '.join(cel.find_all(text=True, recursive=True))), '']
        if col.has_attr('exponent'):
            res[0] = res[0] + ' ' + clean(col['exponent'])
        if col.has_attr('unit'):
            res[1] = clean(col['unit'])
        return res


def findAttachment(col_num, cels, head):
    if col_num == '':
        return ''
    else:
        cel = cels[col_num]
        col = head[col_num]
        # if cel['type'] != col['type']:
        #     print('ERROR\n', col_num, '\n', cels, '\n', head)
        res = clean(' '.join(col.find_all(text=True, recursive=True)))
        return res


def recurFind(res, section, caption, level):
    section = [x for x in section.find_all(level*'sub'+'part') if x.findAll('fig', {'name': caption})]
    if section:
        res.append(section[0]['type'])
        level += 1
        recurFind(res, section[0], caption, level)
    return res


def findSegment(caption, name):
    res = []
    file = open('corpus/'+name+'.xml', 'r', encoding='utf-8')
    soup = BeautifulSoup(file, 'html.parser')
    res = recurFind(res, soup, caption, 0)
    return res


def completeArgs(relation, args):
    df = pd.read_csv(r'input_files\naryrelations.csv', encoding='utf-8')
    candi = df[df.Relation == re.sub(' ', '_', relation.lower())].Argument.tolist()
    for c in candi:
        if c not in args and c != re.sub('_relation', '', re.sub(' ', '_', relation.lower())):
            args[c] = ['', '']
    return args


def parse_tables():
    relation_type = []
    result_arg = []
    args = []
    table_name = []
    caption = []
    segment = []
    document = []

    sought = ['Impact_factor_component_relation',
              'O2 Permeability_Relation',
              'CO2 Permeability_Relation',
              'H2O Permeability_Relation']

    for file in os.listdir('datatables'):
        for table in os.listdir('datatables/'+file):
            if '.html' in table:
                print(table.upper())
                soup = BeautifulSoup(open('datatables/'+file+'/'+table, 'r', encoding='utf8'),
                                     'html.parser')
                head = [x.find(['qc', 'sc']) for x in soup.find('thead').find_all('th')]
                cap = clean(' '.join(soup.find_all('span', {'class': 'captions'})[0].find_all(text=True, recursive=True)))
                for line in soup.find('tbody').find_all('tr'):
                    for relation in line.find_all('ri'):
                        if relation['type'] in sought:
                            ari = relation.find_all('ai')
                            cels = [x if x is not None else BeautifulSoup('<ai type="no" id="no">empty</ai>', 'html.parser') for x in [x.find('ai') for x in line.find_all('td')]]

                            arg_res = [x for x in ari if x['type'] == re.sub('_[Rr]elation', '', relation['type'])]
                            if arg_res:
                                arg_res = arg_res[0]
                                col_num = ''
                                for n in range(0, len(cels)):
                                    if cels[n].text != 'empty':
                                        # print('##########')
                                        # print(cels[n])
                                        # print(arg_res)
                                        # print('##########')
                                        if cels[n]['type'] == arg_res['type'] and cels[n]['id'] == arg_res['id']:
                                            col_num = n
                                arg_resultat = {arg_res['type']: [findValue(col_num, cels, head), findAttachment(col_num, cels, head)]}
                            else:
                                arg_resultat = {re.sub('_[Rr]elation', '', relation['type']): ['', '']}
                            arg_res = re.sub('_[Rr]elation', '', relation['type'])

                            arg_instance = {}
                            for a in ari:
                                if a['type'] != arg_res:
                                    col_num = ''
                                    for n in range(0, len(cels)):
                                        if cels[n].text != 'empty':
                                            if cels[n]['type'] == a['type'] and cels[n]['id'] == a['id']:
                                                col_num = n
                                    arg_instance[head[col_num]['type']] = [findValue(col_num, cels, head), findAttachment(col_num, cels, head)]

                            relation_type.append(relation['type'])
                            result_arg.append(arg_resultat)
                            args.append(completeArgs(relation['type'], arg_instance))
                            table_name.append(re.findall('Table \d+', str(soup))[0])
                            caption.append(cap)
                            segment.append(findSegment(re.findall('Table \d+', str(soup))[0], file))
                            document.append(file)

                        else:
                            continue

    return pd.DataFrame(data={'Relation': relation_type,
                              'Result_Argument': result_arg,
                              'Arguments': args,
                              'Table': table_name,
                              'Caption': caption,
                              'Segment': segment,
                              'Document': document})



if __name__ == '__main__':
    print('Tables Parser')


"""
AJOUTER : traitement des balises aii
"""
