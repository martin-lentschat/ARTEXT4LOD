# coding: utf-8

import os
import re
import ast
import json
import spacy
import stanza
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

import TablesParser


def main():
    if 'resTables.csv' not in os.listdir('datatables'):
        partial_rel = TablesParser.parse_tables()
        partial_rel.to_csv('datatables/resTables.csv', encoding='utf-8')
    partial_rel = pd.read_csv('datatables/resTables.csv', encoding='utf-8')

    instances = pd.read_csv(r'input_files\resFINAL.csv')


"""
#faire fonction qui filtre les instances d'arguments selon les scores de pertinence
"""

if __name__ == '__main__':
    main()
