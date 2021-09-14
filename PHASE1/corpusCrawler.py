# coding: utf-8
from __future__ import unicode_literals

"""
This code takes a set of XML documents and order it for future processing
Be careful that the format have to contains the corrects XML markups
"""

import os
import re
import pandas as pd
import xml.etree.ElementTree as et



def main():
    print('corpusCrawler')


# normalise and clean the textual content
def clean(txt):
    txt = re.sub('\xa0', ' ', txt)
    txt = re.sub(' ?− ?', '-', txt)
    txt = re.sub(' ?– ?', '-', txt)
    txt = re.sub(' ?- ?', '-', txt)
    txt = re.sub(' ?_ ?', ' ', txt)
    txt = re.sub('·', ' ', txt)
    txt = re.sub('/', ' / ', txt)
    txt = re.sub(' ?% ?', '% ', txt)
    txt = re.sub('\n+', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub(' ?± ?', '±', txt)
    txt = re.sub("′", "'", txt)
    txt = re.sub('°', ' °', txt)
    # for x in re.findall('\d-\d', txt):
    #     txt = re.sub(x, x[0] + ' - ' + x[-1], txt)
        # txt[i] = re.sub('', ' ', txt[i])
    txt = re.sub(' +', ' ', txt)
    return txt


# get files from directory
def list_file(loc):
    corpus = os.listdir(loc)
    for i in corpus:
        if '.xml' not in i:
            corpus.remove(i)
    return corpus


# process the XML tree
def parse_file(loc, filename, df):
    tree = et.parse(loc + '/' + filename)
    root = tree.getroot()
    # tag, attrib, parentT, parentA, text = [], [], [], [], []
    level = []
    df = recur(root, df, level, filename)
    return df


def recur(elem, df, level, filename):
    level = level + [elem.get('type')]
    if elem.tag not in ['info', 'content'] and elem.text and re.findall('\w', elem.text):
        txt = clean(elem.text)
        df = df.append({'Document': filename, 'Info': [x for x in level if x], 'Text': txt}, ignore_index=True)
    for i in elem:
        df = recur(i, df, level, filename)
    return df


def parse_all(loc):
    print('####\nDOCs PREPARE ' + str(len(list_file(loc))) + ' files')
    # tag, attrib, section, subsection, text, doc = [], [], [], [], [], []
    df = pd.DataFrame(None, columns=['Document', 'Info', 'Text'])

    for i in list_file(loc):
        df = parse_file(loc, i, df)
        # tag = tag + temp[0]
        # attrib = attrib + temp[1]
        # parentT = parentT + temp[2]
        # parentA = parentA + temp[3]
        # text = text + temp[4]
        # doc = doc + [i] * len(temp[0])
    # df = pd.DataFrame({'Document': doc, 'Tag': tag, 'Attribute': attrib, 'ParentT': parentT, 'ParentA': parentA,
    #                    'Text': clean(text)})
    return df


if __name__ == '__main__':
    main()
