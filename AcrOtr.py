# coding: utf-8
from __future__ import unicode_literals

import re
import os
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords


# dice score between a candidate acronym and the first letters of the considered long form
def calculateScore(acronym, long):
    acronym = re.sub('\\d', '', acronym.lower())
    k = [i[0] for i in re.split('\\W', long) if i]
    s = 0
    p = len(acronym)+len(k)
    for i in acronym:
        if i in k:
            s += 1
            k.remove(i)
    s = 2 * s / p
    return s


# delete the stop-words for score calculation
def cleanSW(t):
    for i in set(stopwords.words('english')):
        t = re.sub(r'\A *?\b'+i+r'\b', '', t)
        t = re.sub(r'\b'+i+r'\b$', '', t)
        t = re.sub(r' +', ' ', t)
    t = t.lower()
    return t


# delete stopwords at the beginning and end
def delSW(t):
    t = nltk.word_tokenize(t)
    while t and t[0] in set(stopwords.words('english')):
        t = t[1:]
    while t and t[-1] in set(stopwords.words('english')):
        t = t[:-1]
    return ' '.join(t)


# get the NP containing a given term out of a sentence
def getNP(w, s):
    np = []
    duo = []
    sent = nltk.word_tokenize(s)
    grammar = 'NP: {<JJ|JJR|JJS|NN|NNS|NNP|NNPS|CC|VBN>*<NN|NNS|NNP|NNPS>+<JJ|JJR|JJS|NN|NNS|NNP|NNPS|CC|VBN>*}'
    patt = nltk.RegexpParser(grammar)
    for i in patt.parse(nltk.pos_tag(sent)).subtrees(filter=lambda t: t.label() == 'NP'):
        if w in ' '.join([x[0] for x in i.leaves()]):
            np.append(' '.join([x[0] for x in i.leaves()]))

    for i in re.findall('\\(.+?\\)', s):
        for j in np:
            j = delSW(j)
            score = calculateScore(i[1:-1], cleanSW(j))
            duo.append([j, i[1:-1], score])
    return duo


# extract a list of [long form , acronym, similarity score] for a given vocabulary in a corpus
def getCandidates(corp, voca, fastr):
    kind, node, term, duo, phrase = [], [], [], [], []
    for p in tqdm(corp.Text.values.tolist()):
        sentences = nltk.sent_tokenize(p)
        for sent in sentences:
            for i in voca.itertuples():
                for j in i.PrefLabel:
                    if j in sent and len(j) > 3:
                        kind.append('PrefLabel')
                        node.append(i.Node)
                        term.append(j)
                        duo.append(getNP(j, sent))
                        phrase.append(sent)
                for j in i.AltLabel:
                    if j in sent and len(j) > 3:
                        kind.append('AltLabel')
                        node.append(i.Node)
                        term.append(j)
                        duo.append(getNP(j, sent))
                        phrase.append(sent)
                if fastr:
                    for j in i.FastrPref:
                        if j in sent and len(j) > 3:
                            kind.append('FastrPref')
                            node.append(i.Node)
                            term.append(j)
                            duo.append(getNP(j, sent))
                            phrase.append(sent)
                    for j in i.FastrAlt:
                        if j in sent and len(j) > 3:
                            kind.append('FastrAlt')
                            node.append(i.Node)
                            term.append(j)
                            duo.append(getNP(j, sent))
                            phrase.append(sent)
    df = pd.DataFrame({'Type': kind, 'Node': node, 'Term': term, 'Duo': duo, 'Sentence': phrase})
    return df


# select the acronyms over a given threshold and get rid of duplicates
def reduceDF(df, threshold):
    print('reduction ...')
    kind, node, term, long, acro, score, phrase = [], [], [], [], [], [], []
    for i in df.itertuples():
        duodf = i.Duo
        for d in duodf:
            if d[-1] > threshold and len(d[1]) > 1:
                kind.append(i.Type)
                node.append(i.Node)
                term.append(i.Term)
                long.append(d[0])
                acro.append(d[1])
                score.append(d[-1])
                phrase.append(i.Sentence)
    df2 = pd.DataFrame({'Type': kind, 'Node': node, 'Term': term, 'LongForm': long, 'Acronym': acro, 'Score': score,
                        'Sentence': phrase}
                       ).sort_values(['Score'], ascending=False).drop_duplicates(['LongForm'], keep='first')
    return df2


# add its acronyms to each node in the OTR vocabulary
def addAcroFind(voca, corp, fastr, threshold):
    voca['AcroPrefLabel'] = voca.apply(lambda x: [], axis=1)
    voca['AcroAltLabel'] = voca.apply(lambda x: [], axis=1)
    if fastr:
        voca['AcroFastrPref'] = voca.apply(lambda x: [], axis=1)
        voca['AcroFastrAlt'] = voca.apply(lambda x: [], axis=1)
    if 'acronymKept.csv' not in os.listdir('work_files/'):
        found = getCandidates(corp, voca, fastr)
        found.to_csv('work_files/acronymFound.csv', sep=',', encoding='utf-8')
        kept = reduceDF(found, threshold)
        kept.to_csv('work_files/acronymKept.csv', sep=',', encoding='utf-8')
    else:
        kept = pd.read_csv('work_files/acronymKept.csv')
    for i in voca.itertuples():
        for j in kept.itertuples():
            if i.Node == j.Node:
                voca.at[i.Index, 'Acro' + j.Type] = voca.at[i.Index, 'Acro'+j.Type]+[[j.Acronym, j.LongForm, j.Score]]
    return voca


def main():
    print('AcrOtr')


if __name__ == '__main__':
    main()
