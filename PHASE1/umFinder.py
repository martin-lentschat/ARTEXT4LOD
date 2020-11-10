# coding: utf-8
from __future__ import unicode_literals

"""
Find measure unit variations
"""
import os
import re
import nltk
import distance
import pandas as pd
from tqdm import tqdm
from nltk.corpus import words


def jaccard_indice(list1, list2):
    return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))


def similarity_extended(list1, list2):
    return max(0.0, (min(len(list1), len(list2)) - distance.levenshtein(list1, list2)) / (
        min(len(list1), len(list2))))


def cleanUM(x):
    x = re.sub(r'\^', '', x)
    x = re.sub(r'-', '−', x)
    x = re.sub(r'µ', 'μ', x)
    # x = re.sub(r'\^', '', x)
    return x


# find a unit tocken, expand to the all unit and compute similarity to existing ones
def find_all_units(corpus, um):
    if 'unitsVar.csv' not in os.listdir('work_files/'):
        units = []
        cleara = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ' ', ',', '±', '–', '×',
                  '^', '-', ';', '(']
        clearb = ['.', ' ', ',', '±', '-', '×', ';', ')']
        for i in tqdm(range(0, len(corpus))):
            txt = corpus.Text.values[i]
            sent = nltk.sent_tokenize(txt)
            for s in sent:
                tok = nltk.word_tokenize(s)
                for t in range(0, len(tok)):
                    if tok[t].replace('^', '') in um and re.findall('[^0-9]',
                                                                    tok[t].replace('^', '')):
                        indice = t + 1
                        unit = tok[t]
                        while indice < len(tok) and \
                                (tok[indice].lower() not in words.words() and len(
                                    tok[indice].lower()) > 1):
                            unit = unit + ' ' + tok[indice]
                            indice += 1
                        indice = t - 1
                        while indice > 0 and \
                                (tok[indice].lower() not in words.words() and len(
                                    tok[indice].lower()) > 1):
                            unit = tok[indice] + ' ' + unit
                            indice -= 1
                        while unit and unit[0] in cleara:
                            unit = unit[1:]
                        while unit and unit[-1] in clearb:
                            unit = unit[:-1]
                        if unit not in units and unit not in um:
                            units.append(unit)
        variants = {}
        for i in units:
            a = re.split(r'[. /\\()]', cleanUM(i))
            variants[i] = []
            for j in um:
                b = re.split(r'[. /\\]', cleanUM(j))
                if jaccard_indice(a, b) >= 0.4:
                    if similarity_extended(a, b) >= 0.4 \
                            and [jaccard_indice(a, b), j, similarity_extended(a, b)] not in \
                            variants[i]:
                        variants[i].append([similarity_extended(a, b), j, jaccard_indice(a, b)])
        df = pd.DataFrame(None, columns=['UMnew', 'UMrto', 'Scores'])
        for i in variants:
            if variants[i]:
                df = df.append({'UMnew': i,
                                'UMrto': max(variants[i])[1],
                                'Scores': (max(variants[i])[0], max(variants[i])[2])},
                               ignore_index=True)
    else:
        df = pd.read_csv('work_files/unitsVar.csv', encoding='utf-8')
        del df['Unnamed: 0']
    return df


def main():
    print('umFinder')


if __name__ == '__main__':
    main()
