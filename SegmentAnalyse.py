# coding: utf-8
from __future__ import unicode_literals

"""
"""
import re
import os
import ast
import math
import json
import spacy
import stanza
import Levenshtein
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords


# clean the section names
def simple(s):
    s = s.lower()
    s = re.sub('[([{].{1,10}[)\\]}]', '', s, re.DOTALL)
    return s


# create the Segment classes
def classCreation(titles):
    classes = {}
    for i in titles:
        if i in classes:
            classes[i][0] += 1
        else:
            new = True
            for j in classes:
                if Levenshtein.distance(simple(j), simple(i)) < 4:
                    classes[j][0] += 1
                    if i not in classes[j][1]:
                        classes[j][1].append(i)
                    new = False
            if new:
                classes[i] = [1, [i]]
    temp = dict(classes)
    for i in temp:
        for j in classes:
            poped = False
            for k in classes[j][1]:
                if i.lower() in k.lower() and i != j:
                    poped = True
                    classes[j][0] += temp[i][0]
                    for t in temp[i][1]:
                        classes[j][1].append(t)
                    classes.pop(i)
                    break
            if poped:
                break
    return classes


# create bags of words for each document
def createBagD(corp, docs, tokenicer, sentencer):
    bag = {}
    for d in tqdm(docs):
        text_content = ''
        for c in corp.itertuples():
            if c.Document == d:
                text_content += c.Text
        words = []
        for s in [x.text for x in sentencer(text_content).sentences]:
            words = words + [w.text for w in tokenicer(s)
                             if len(w.text) > 1 and w.text.lower() not in stopwords.words('english')]
        bag[d] = (len(words), text_content.lower())
    return bag


# create bags of words for each class
def createBag(corp, classes, tokenicer, sentencer):
    bag = {}
    for c in tqdm(classes):
        text_content = ''
        for d in corp.itertuples():
            if list(set(ast.literal_eval(d.Info)) & set(classes[c][1])):
                text_content += d.Text
        words = []
        for s in [x.text for x in sentencer(text_content).sentences]:
            words = words + [w.text for w in tokenicer(s)
                             if len(w.text) > 1 and w.text.lower() not in stopwords.words('english')]
        bag[c] = (len(words), text_content.lower())
    return bag


def get_vocabulary(arg):
    vocarg = {}
    for a in arg.itertuples():
        if a.Argument in vocarg:
            vocarg[a.Argument] += [w.lower() for w in ast.literal_eval(a.Disambiguation)[0]
                                   if len(w) > 1 and w.lower() not in stopwords.words('english')]
        else:
            vocarg[a.Argument] = [w.lower() for w in ast.literal_eval(a.Disambiguation)[0]
                                  if len(w) > 1 and w.lower() not in stopwords.words('english')]
    for a in vocarg:
        vocarg[a] = list(set(vocarg[a]))
    return vocarg


# compute the inverse category frequency scores for a set of multi-words term
def get_if(terms, bags):
    n = 0
    for b in bags:
        for t in terms:
            if t in bags[b][1]:
                n += 1
                break
    if n == 0:
        print('IF=0', terms)
    else:
        return math.log10(len(bags.keys()) / n)


# compute the inverse category frequency scores for a set of multi-words term
def get_tf(terms, bag):
    freq = sum([len(re.findall(re.escape(x), bag[1])) for x in terms])
    if freq == 0:
        print('TF=0', terms)
    return freq / bag[0]


# compute all of the contexts scores of the Segments on a given level
def contextualise(classes, corp, level):
    if level+'segment_bags.json' not in os.listdir('work_files/'):
        tokenicer = spacy.load("en_core_web_sm")
        sentencer = stanza.Pipeline(processors='tokenize', lang='en', package='partut')
        print('Segment scores  -> creat bags '+level)
        bags = createBag(corp, classes, tokenicer, sentencer)
        open('work_files/'+level+'segment_bags.json', 'w', encoding='utf-8').write(json.dumps(bags, indent=4))


def classic(corp):
    if 'bags_document.json' not in os.listdir('work_files/'):
        tokenicer = spacy.load("en_core_web_sm")
        sentencer = stanza.Pipeline(processors='tokenize', lang='en', package='partut')
        print('Classic Scores -> creat bags')
        bags = createBagD(corp, list(set(corp.Document.values)), tokenicer, sentencer)
        open('work_files/bags_document.json', 'w', encoding='utf-8').write(json.dumps(bags, indent=4))


# all process of creating the Segment classes and then computing the lexical scores
def prepare_Scores(arg, corp):
    titles_top = [ast.literal_eval(x)[0] for x in arg.drop_duplicates(['Segment', 'Document'], keep='first').Segment.values]
    titles_bottom = [ast.literal_eval(x)[-1] for x in arg.drop_duplicates(['Segment', 'Document'], keep='first').Segment.values]
    if 'TOPsegment-Classes.json' not in os.listdir('work_files/'):
        classes = classCreation(titles_top)
        open('work_files/TOPsegment-Classes.json', 'w', encoding='utf-8').write(json.dumps(classes, indent=4))
    else:
        with open('work_files/TOPsegment-Classes.json') as f:
            classes = json.load(f)
    contextualise(classes, corp, 'TOP')
    if 'BOTsegment-Classes.json' not in os.listdir('work_files/'):
        classes = classCreation(titles_bottom)
        open('work_files/BOTsegment-Classes.json', 'w', encoding='utf-8').write(json.dumps(classes, indent=4))
    else:
        with open('work_files/BOTsegment-Classes.json') as f:
            classes = json.load(f)
    contextualise(classes, corp, 'BOT')
    classic(corp)


# add the context scores to the arguments
def add_Scores(arg):
    res = pd.DataFrame(None, ['Argument', 'Track', 'Span', 'Disambiguation', 'Sentence', 'Window', 'Segment',
                              'Document',
                              'DC_Tree',
                              'TF_segment_term_top', 'ICF_segment_term_top', 'TF_segment_arg_top', 'ICF_segment_arg_top',
                              'TF_segment_term_bot', 'ICF_segment_term_bot', 'TF_segment_arg_bot', 'ICF_segment_arg_bot',
                              'TF_classic_term', 'IDF_classic_term', 'TF_classic_arg', 'IDF_classic_arg'])

    vocarg = get_vocabulary(arg)
    with open('work_files/TOPsegment-Classes.json') as f:
        classes_top = json.load(f)
    with open('work_files/BOTsegment-Classes.json') as f:
        classes_bot = json.load(f)
    with open('work_files/TOPsegment_bags.json') as f:
        bags_segment_top = json.load(f)
    with open('work_files/BOTsegment_bags.json') as f:
        bags_segment_bot = json.load(f)
    with open('work_files/bags_document.json') as f:
        bags_document = json.load(f)

    tree_depths = {}
    for i in set(arg.Argument.values):
        tree_depths[i] = list(set([ast.literal_eval(x)[1] + 1 for x in list(set(arg[arg.Argument == i].Track.values))]))

    for a in tqdm(arg.itertuples()):
        words = [w.lower() for w in ast.literal_eval(a.Disambiguation)[0]
                 if len(w) > 1 and w.lower() not in stopwords.words('english')]

        classe_top = [x for x in classes_top.keys() if ast.literal_eval(a.Segment)[0] in classes_top[x][1]][0]
        classe_bot = [x for x in classes_bot.keys() if ast.literal_eval(a.Segment)[-1] in classes_bot[x][1]][0]
        argument = a.Argument
        doc = a.Document

        res = res.append({'Argument': argument,
                          'Track': a.Track,
                          'Span': a.Span,
                          'Disambiguation': a.Disambiguation,
                          'Sentence': a.Sentence,
                          'Window': a.Window,
                          'Segment': a.Segment,
                          'Document': doc,

                          'DC_Tree': (ast.literal_eval(a.Track)[1] + 1) / (max(tree_depths[argument]) + 1),

                          'TF_segment_term_top': get_tf(words, bags_segment_top[classe_top]),
                          'ICF_segment_term_top': get_if(words, bags_segment_top),
                          'TF_segment_arg_top': get_tf(vocarg[argument], bags_segment_top[classe_top]),
                          'ICF_segment_arg_top': get_if(vocarg[argument], bags_segment_top),

                          'TF_segment_term_bot': get_tf(words, bags_segment_bot[classe_bot]),
                          'ICF_segment_term_bot': get_if(words, bags_segment_bot),
                          'TF_segment_arg_bot': get_tf(vocarg[argument], bags_segment_bot[classe_bot]),
                          'ICF_segment_arg_bot': get_if(vocarg[argument], bags_segment_bot),

                          'TF_classic_term': get_tf(words, bags_document[doc]),
                          'IDF_classic_term': get_if(words, bags_document),
                          # 'TF_classic_arg': get_tf(vocarg[argument], bags_document[doc]),
                          # 'IDF_classic_arg': get_if(vocarg[argument], bags_document)
                          }, ignore_index=True, sort=False)
    return res


def main():
    print('segment analyse')


if __name__ == '__main__':
    main()
