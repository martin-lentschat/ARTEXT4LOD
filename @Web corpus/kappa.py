# coding: utf-8
from __future__ import unicode_literals

"""
"""

import re
import os
import ast
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def clean(txt):
    txt = re.sub('−', '-', txt)
    txt = re.sub('–', '-', txt)
    txt = re.sub("′", "'", txt)
    txt = re.sub('\xa0', ' ', txt)
    txt = re.sub(' +', '', txt)
    return txt.lower()


def correspondence_arg(arg):
    correspondence = {'CO2\\_Permeability': 'Permeability',
                      'O2\\_Permeability': 'Permeability',
                      'H2O\\_Permeability': 'Permeability',
                      'Method': 'Method',
                      'measure\\_Unit': 'measure unit',
                      'Thickness': 'Thickness',
                      'numeric\\_Value': 'numeric value',
                      'Temperature': 'Temperature',
                      'Relative\\_Humidity\\_Difference': 'Relative Humidity',
                      'Relative\\_Humidity': 'Relative Humidity',
                      'Packaging\\_component': 'Component',
                      'Partial\\_Pressure\\_Difference': 'Partial Pressure',
                      'Packaging\\_name': 'Packaging'}
    return correspondence[arg]


def main():
    for g1, g2 in [('golden-anno1', 'golden-anno2'), ('golden-anno1', 'golden-anno3')]:
        print('##### ' + g1 + ' -> KAPPA <- ' + g2 + ' #####')
        annotations1 = {}
        annotations2 = {}
        for file in os.listdir(r'golden/' + g1 + '/'):
            if '.tsv' in file:
                with open(r'golden/' + g1 + '/' + file, encoding='utf-8') as f:
                    file1 = csv.reader(f, delimiter="\t")
                    for line in file1:
                        if len(line) > 1 and line[3] != '_':
                            value = clean(line[2])
                            if value:
                                for c in line[3].split('|'):
                                    catego = correspondence_arg(c.split('[')[0])
                                    if catego not in annotations1:
                                        annotations1[catego] = [value]
                                    elif value not in annotations1[catego]:
                                        annotations1[catego] = annotations1[catego] + [value]
                with open(r'golden/' + g2 + '/' + file, encoding='utf-8') as f:
                    file2 = csv.reader(f, delimiter="\t")
                    for line in file2:
                        if len(line) > 1 and line[3] not in ['', '_']:
                            value = clean(line[2])
                            if value:
                                for c in line[3].split('|'):
                                    catego = correspondence_arg(c.split('[')[0])
                                    if catego not in annotations2:
                                        annotations2[catego] = [value]
                                    elif value not in annotations2[catego]:
                                        annotations2[catego] = annotations2[catego] + [value]

        categories = list(set(list(set(annotations1.keys())) + list(set(annotations2.keys()))))
        matrix = np.array(np.empty((len(categories), len(categories)), object))
        for c1 in range(0, len(categories)):
            if categories[c1] not in annotations1.keys():
                continue
            cate1 = categories[c1]
            for a1 in annotations1[cate1]:
                for c2 in range(0, len(categories)):
                    if categories[c2] not in annotations2.keys():
                        continue
                    cate2 = categories[c2]
                    if a1 in annotations2[cate2]:
                        if not matrix[c1][c2]:
                            matrix[c1][c2] = [a1]
                        elif a1 not in matrix[c1][c2]:
                            matrix[c1][c2].append(a1)
                        else:
                            matrix[c1][c2].append(a1)
        for c2 in range(0, len(categories)):
            if categories[c2] not in annotations2.keys():
                continue
            cate2 = categories[c2]
            for a2 in annotations2[cate2]:
                for c1 in range(0, len(categories)):
                    if categories[c1] not in annotations1.keys():
                        continue
                    cate1 = categories[c1]
                    if a2 in annotations1[cate1]:
                        if not matrix[c2][c1]:
                            matrix[c2][c1] = [a2]
                        elif a2 not in matrix[c2][c1]:
                            matrix[c2][c1].append(a2)
                        else:
                            matrix[c2][c1].append(a2)

        matrix_n = matrix
        for i in range(0, len(categories)):
            for j in range(0, len(categories)):
                if not matrix_n[i][j]:
                    matrix_n[i][j] = 0
                else:
                    matrix_n[i][j] = len(matrix_n[i][j])

        len_tot = matrix_n.sum()
        print(len_tot)
        commun = 0
        for i in range(0, len(matrix_n)):
            commun += matrix_n[i][i]
        accord = commun / len_tot
        print(accord)
        hasard = 0
        for i in range(0, len(matrix_n)):
            hasard += (sum(matrix_n[i] / len_tot)) * (sum(matrix_n[:, i]) / len_tot)
        print(hasard)
        kappa = (accord - hasard) / (1 - hasard)
        print(kappa)


if __name__ == '__main__':
    main()
