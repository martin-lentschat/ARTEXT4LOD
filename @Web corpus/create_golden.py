# coding: utf-8
from __future__ import unicode_literals

"""
"""

import os
import csv
import pandas as pd
from tqdm import tqdm


def find_all(content, item):
    if '[' in item[2][0]:
        return [z for z in content if item[2][0] in z[2]]
    else:
        return [item]


def create_golden(path):
    res = pd.DataFrame(None, columns=['ID', 'Doc', 'Argument', 'NumValUM', 'Token'])
    ide = 0
    for i in tqdm(os.listdir(path)):
        content = []
        # print(i)
        with open(path + i, encoding='utf-8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            file = []
            for r in tsv_reader:
                file.append(r)
            for row in range(0, len(file)):
                if len(file[row]) > 1:
                    if file[row][-4] != '_':
                        content.append([file[row][0], file[row][2], file[row][-4].split('|'), file[row][-3].split('|'),
                                        file[row][-2].split('|')])
        for c in content:  # Perm, RH, Thick, PPP, (some)Temperature
            if 'isUnitOf' in c[3]:
                arg = c
                argE = find_all(content, arg)
                for x in range(0, len(arg[3])):
                    unit = [z for z in content if arg[4][x].split('[')[0] in z[0]][0]
                    if unit[3] == ['_']:
                        continue
                    unitE = find_all(content, unit)
                    for y in range(0, len(unit[3])):
                        num = [z for z in content if unit[4][y].split('[')[0] in z[0]][0]
                        numE = find_all(content, num)
                        res, ide = add_row(res, ide, i, argE[0][2][0], ([x[1] for x in numE if x[1] is not '\xa0'],
                                                                        [x[1] for x in unitE if x[1] is not '\xa0']),
                                           [x[1] for x in argE])
        for c in content:  # (most)Temperature
            unitE = []
            if ('°' or ' ') in c[1] and 'measure\\_Unit' in c[2][0]:
                if c in unitE:
                    continue
                unitE = find_all(content, c)
                for x in range(0, len(unitE[0][3])):
                    num = [z for z in content if unitE[0][4][x].split('[')[0] in z[0]][0]
                    numE = find_all(content, num)
                    res, ide = add_row(res, ide, i, '5', ([x[1] for x in numE if x[1] is not '\xa0'],
                                                          [x[1] for x in unitE if x[1] is not '\xa0']), '')
        for c in content:  # Packaging component -> impact factor
            if 'isCompositionValueOf' in c[3]:
                arg = c
                argE = find_all(content, arg)
                for x in range(0, len(arg[3])):
                    unit = [z for z in content if arg[4][x].split('[')[0] in z[0]][0]
                    unitE = find_all(content, unit)
                    if unit[3] == ['_']:
                        res, ide = add_row(res, ide, i, '4', ([x[1] for x in unitE], ''), [x[1] for x in argE])
                        continue
                    for y in range(0, len(unit[3])):
                        num = [z for z in content if unit[4][y].split('[')[0] in z[0]][0]
                        numE = find_all(content, num)
                        res, ide = add_row(res, ide, i, '4', ([x[1] for x in numE if x[1] is not '\xa0'],
                                                              [x[1] for x in unitE if x[1] is not '\xa0']),
                                           [x[1] for x in argE])
        for c in content:  # Packaging Name, Packaging Component, Method
            for x in c[2]:
                if r'Packaging\_name' in x:
                    argE = find_all(content, c)
                    res, ide = add_row(res, ide, i, '1', [x[1] for x in argE], '')
                if r'Packaging\_component' in x:
                    argE = find_all(content, c)
                    if 'isCompositionValueOf' not in [y for x in argE for y in x[3]]:
                        res, ide = add_row(res, ide, i, '2', [x[1] for x in argE], '')
                if r'Method' in x:
                    argE = find_all(content, c)
                    res, ide = add_row(res, ide, i, '3', [x[1] for x in argE], '')
    return res


def add_row(df, ide, doc, arg, val, token):
    cores = {'1': ('Packaging Name', 'SYMBOLIC'), '2': ('Packaging Component', 'SYMBOLIC'), '3': ('Method', 'SYMBOLIC'),
             '4': ('Impact Factor', 'ADDIMENTIONNAL'), '5': ('Temperature', 'QUANTITY'),
             'Temperature': ('Temperature', 'QUANTITY'),
             r'Partial\_Pressure\_Difference': ('Partial Pressure Difference', 'QUANTITY'),
             r'Thickness': ('Thickness', 'QUANTITY'),
             r'Relative\_Humidity': ('Relative Humidity', 'QUANTITY'),
             r'Relative\_Humidity\_Difference': ('Relative Humidity', 'QUANTITY'),
             r'O2\_Permeability': ('Oxygen Permeability', 'QUANTITY'),
             r'H2O\_Permeability': ('Water Permeability', 'QUANTITY'),
             r'CO2\_Permeability': ('Carbon Dioxyde Permeability', 'QUANTITY')}
    df = df.append({'ID': ide,
                    'Doc': ''.join(doc.split('.')[0:-2]),
                    'Argument': correspondence_arg(cores[arg.split('[')[0]][0]),
                    'Category': cores[arg.split('[')[0]][1],
                    'NumValUM': val,
                    'Token': token},
                   ignore_index=True)
    ide += 1
    return df, ide


def correspondence_arg(arg):
    correspondence = {'Relative Humidity Difference': 'relative humidity difference',
                      'Packaging Component': 'impact_factor_component',
                      'Method': 'method',
                      'Water Permeability': 'h2o_permeability',
                      'Oxygen Permeability': 'o2_permeability',
                      'Partial Pressure Difference': 'partial_pressure_difference',
                      'Carbon Dioxyde Permeability': 'co2_permeability',
                      'Thickness': 'thickness',
                      'Relative Humidity': 'relative_humidity',
                      'Temperature': 'temperature',
                      'Impact Factor': 'component_qty_value',
                      'Packaging Name': 'packaging'}
    return correspondence[arg]


def main():
    pd.options.mode.chained_assignment = None
    for g in os.listdir('golden/'):
        if g not in ['golden-all']:
            break
        print('##### ' + g.upper() + ' #####')
        if g + 'BIG.csv' not in os.listdir('golden/' + g + '/'):
            print('#' + g.upper())
            golden = create_golden('golden/' + g + '/')
            golden.to_csv('golden/' + g + '/' + g + 'BIG.csv', encoding='utf-8')
            golden = pd.read_csv('golden/' + g + '/' + g + 'BIG.csv', encoding='utf-8')
            golden.drop_duplicates(['Doc', 'Argument', 'NumValUM'], keep='first') \
                .to_csv('golden/' + g + '/' + g + 'LIGHT.csv', encoding='utf-8')


if __name__ == '__main__':
    main()
