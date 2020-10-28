# coding: utf-8
from __future__ import unicode_literals

import os
import re
import ast
import json
import spacy
import stanza
import warnings
import pandas as pd
from tqdm import tqdm
from spacy.symbols import ORTH

import AcrOtr
import matcher
import umFinder
import ontoCrawler
import corpusCrawler
import SegmentAnalyse
import DrivenExtraction


# manually add some vocabulary
def addManuVoc(df):
    manual = {'co2_permeability': ['CO2', 'CO[2]'],
              'o2_permeability': ['O[2]P', 'PO[2]', 'PO2', 'O2', 'OTR', 'oxygen transmission rate'],
              'h2o_permeability': ['H2O', 'WTR', 'WVP', 'water-vapor transmission rate'],
              'partial_pressure_difference': ['PP', 'P[0]', 'difference in pressure'],
              'relative_humidity': ['RH', 'ambient humidity']}
    df['ManuLabel'] = df.apply(lambda x: [], axis=1)
    for i in df.itertuples():
        if i.Node in manual.keys():
            for k in manual[i.Node]:
                if k not in allVoc(i):
                    df.at[i.Index, 'ManuLabel'] = df.loc[i.Index, 'ManuLabel'] + [k]
    return df


# return a list of all the terms from a df row
def allVoc(row):
    terms = ast.literal_eval(row.PrefLabel) if type(row.PrefLabel) == str else row.PrefLabel
    if row.AltLabel:
        terms = terms + ast.literal_eval(row.AltLabel) if type(row.AltLabel) == str else row.AltLabel
    return list(set(terms))


# launch FASTR if necessary and add the vocabulary variations
def fastring(df):
    df['FastrPref'] = df.apply(lambda x: [], axis=1)
    df['FastrAlt'] = df.apply(lambda x: [], axis=1)
    df['FastrManu'] = df.apply(lambda x: [], axis=1)
    if 'PrefLabels_driven_extraction.json' in os.listdir('work_files/'):
        with open('work_files/PrefLabels_driven_extraction.json') as json_file:
            js = json.load(json_file)
    else:
        js = DrivenExtraction.main2('PrefLabels', 'corpus', [y for x in df.PrefLabel.values.tolist() for y in x],
                                    2, False, True, False)
    for i in df.itertuples():
        for j in js:
            if j in i.PrefLabel:
                df.at[i.Index, 'FastrPref'] = df.at[i.Index, 'FastrPref'] + js[j]
    if 'AltLabels_driven_extraction.json' in os.listdir('work_files/'):
        with open('work_files/AltLabels_driven_extraction.json') as json_file:
            js = json.load(json_file)
    else:
        js = DrivenExtraction.main2('AltLabels', 'corpus', [y for x in df.AltLabel.values.tolist() for y in x],
                                    2, False, True, False)
    for i in df.itertuples():
        for j in js:
            if j in i.AltLabel:
                df.at[i.Index, 'FastrAlt'] = df.at[i.Index, 'FastrAlt'] + js[j]
    if 'ManuLabels_driven_extraction.json' in os.listdir('work_files/'):
        with open('work_files/ManuLabels_driven_extraction.json') as json_file:
            js = json.load(json_file)
    else:
        js = DrivenExtraction.main2('ManuLabels', 'corpus', [y for x in df.ManuLabel.values.tolist() for y in x],
                                    2, False, True, False)
    for i in df.itertuples():
        for j in js:
            if j in i.ManuLabel:
                df.at[i.Index, 'FastrManu'] = df.at[i.Index, 'FastrManu'] + js[j]
    return df  # .drop_duplicates(['Concept', 'Term'], keep='first')


# create the data frame for the csv unit file
def createUnitFile(base, found, desambig):
    pref = []
    alt = []
    var = []
    asso = []
    if 'UnitsLabels_driven_extraction.json' in os.listdir('work_files/'):
        with open('work_files/UnitsLabels_driven_extraction.json') as json_file:
            js = json.load(json_file)
    else:
        js = DrivenExtraction.main2('UnitsLabels', 'corpus', list(set(base.PrefLabel.values.tolist())), 2, False, False,
                                    False)
    for i in base.itertuples():
        if i.PrefLabel not in pref:
            pref.append(i.PrefLabel)
            alt.append(list(set([x for x in base[base.PrefLabel == i.PrefLabel].AltLabel.values.tolist()])))
            temp = []
            for j in [x for x in base[base.PrefLabel == pref[-1]].AltLabel.values.tolist()]:
                if found[found.UMrto == j].UMnew.values.tolist():
                    temp.append(found[found.UMrto == j].UMnew.values.tolist()[0])
                if i.PrefLabel in js:
                    for x in js[i.PrefLabel]:
                        temp.append(x)
            var.append(list(set(temp)))
            temp = []
            for j in desambig[desambig.Unit == pref[-1]].Concept.values.tolist():
                temp.append(j.split('#')[-1])
            for j in alt[-1]:
                for k in desambig[desambig.Unit == j].Concept.values.tolist():
                    temp.append(k.split('#')[-1])
            asso.append(list(set(temp)))
    df = pd.DataFrame({'PrefLabel': pref, 'AltLabel': alt, 'VarLabel': var, 'Associations': asso})
    return df


def build_um(um, nlp):
    res = {}
    for i in um.itertuples():
        for t in i.PrefLabel.split(' '):
            nlp.tokenizer.add_special_case(t, [{ORTH: t}])
        res[i.PrefLabel] = [i.Associations, [x.text for x in nlp(i.PrefLabel)]]
        if i.AltLabel:
            for j in i.AltLabel:
                for t in j.split(' '):
                    nlp.tokenizer.add_special_case(t, [{ORTH: t}])
                res[j] = [i.Associations, [x.text for x in nlp(j)]]
        if i.VarLabel:
            for j in i.VarLabel:
                for t in j.split(' '):
                    nlp.tokenizer.add_special_case(t, [{ORTH: t}])
                res[j] = [i.Associations, [x.text for x in nlp(j)]]
    open('work_files/units.json', 'w', encoding='utf-8').write(json.dumps(res, indent=4, ensure_ascii=False))
    return res, nlp


def build_terms(voca, nlp):
    res = {}
    for i in voca.itertuples():
        for j in i.PrefLabel:
            # if i.Argument + i.Node + 'PrefLabel' + j not in res:
            for t in j.split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            res[i.Argument + i.Node + 'PrefLabel' + j] = [i.Argument, i.Node, i.Depth, 'PrefLabel',
                                                          [x.text for x in nlp(j)]]
        for j in i.AltLabel:
            for t in j.split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AltLabel' not in res:
            res[i.Argument + i.Node + 'AltLabel' + j] = [i.Argument, i.Node, i.Depth, 'AltLabel',
                                                         [x.text for x in nlp(j)]]
        for j in i.FastrPref:
            for t in j.split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'FastrPref' not in res:
            res[i.Argument + i.Node + 'FastrPref' + j] = [i.Argument, i.Node, i.Depth, 'FastrPref',
                                                          [x.text for x in nlp(j)]]
        for j in i.FastrAlt:
            for t in j.split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'FastrAlt' not in res:
            res[i.Argument + i.Node + 'FastrAlt' + j] = [i.Argument, i.Node, i.Depth, 'FastrAlt',
                                                         [x.text for x in nlp(j)]]
        for j in i.AcroPrefLabel:
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroPrefLabel-0' not in res:
            res[i.Argument + i.Node + 'AcroPrefLabel-0' + j[0]] = [i.Argument, i.Node, i.Depth, 'AcroPrefLabel-0',
                                                                   [x.text for x in nlp(j[0])]]
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroPrefLabel-1' not in res:
            res[i.Argument + i.Node + 'AcroPrefLabel-1' + j[1]] = [i.Argument, i.Node, i.Depth, 'AcroPrefLabel-1',
                                                                   [x.text for x in nlp(j[1])]]
        for j in i.AcroAltLabel:
            # if i.Argument + i.Node + 'AcroAltLabel-0' not in res:
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            res[i.Argument + i.Node + 'AcroAltLabel-0' + j[0]] = [i.Argument, i.Node, i.Depth, 'AcroAltLabel-0',
                                                                  [x.text for x in nlp(j[0])]]
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroAltLabel-1' not in res:
            res[i.Argument + i.Node + 'AcroAltLabel-1' + j[1]] = [i.Argument, i.Node, i.Depth, 'AcroAltLabel-1',
                                                                  [x.text for x in nlp(j[1])]]
        for j in i.AcroFastrPref:
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroFastrPref-0' not in res:
            res[i.Argument + i.Node + 'AcroFastrPref-0' + j[0]] = [i.Argument, i.Node, i.Depth, 'AcroFastrPref-0',
                                                                   [x.text for x in nlp(j[0])]]
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroFastrPref-1' not in res:
            res[i.Argument + i.Node + 'AcroFastrPref-1' + j[1]] = [i.Argument, i.Node, i.Depth, 'AcroFastrPref-1',
                                                                   [x.text for x in nlp(j[1])]]
        for j in i.AcroFastrAlt:
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroFastrAlt-0' not in res:
            res[i.Argument + i.Node + 'AcroFastrAlt-0' + j[0]] = [i.Argument, i.Node, i.Depth, 'AcroFastrAlt-0',
                                                                  [x.text for x in nlp(j[0])]]
            for t in j[0].split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'AcroFastrAlt-1' not in res:
            res[i.Argument + i.Node + 'AcroFastrAlt-1' + j[1]] = [i.Argument, i.Node, i.Depth, 'AcroFastrAlt-1',
                                                                  [x.text for x in nlp(j[1])]]
        for j in i.ManuLabel:
            for t in j.split(' '):
                nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            # if i.Argument + i.Node + 'ManuLabel' not in res:
            res[i.Argument + i.Node + 'ManuLabel' + j] = [i.Argument, i.Node, i.Depth, 'ManuLabel',
                                                          [x.text for x in nlp(j)]]
    open('work_files/terms.json', 'w', encoding='utf-8').write(json.dumps(res, indent=4, ensure_ascii=False))
    return res, nlp


# extract the windows from the documents with their respective entities of interest
def extractWindow(corp, um, voca, debut, fin, noLimit):

    print('#BUILD TOKENIZER ... ', end='')
    tokenicer = spacy.load("en_core_web_sm")
    refsUM, tokenicer = build_um(um, tokenicer)
    print('units ready ... ', end='')
    refsTerms, tokenicer = build_terms(voca, tokenicer)
    print('terms ready')

    win = pd.DataFrame(None, columns=['UM', 'NumVal', 'Terms', 'Segment', 'Text', 'Doc'])
    text = corp.Text.values
    info = corp.Info.values
    file = corp.Document.values

    sentencer = stanza.Pipeline(processors='tokenize', lang='en', package='partut')
    for i in tqdm(range(0, len(corp))):
        sentences = [x.text for x in sentencer(text[i]).sentences]
        for s in range(0, len(sentences)):
            sent = []
            for x in range(debut, fin + 1):
                if 0 <= s + x < len(sentences):
                    sent.append([x.text for x in tokenicer(sentences[s + x])])
                else:
                    sent.append('')
            units = []
            values = []
            terms = []
            for q in range(0, len(sent)):
                if sent[q]:
                    t = findTerms(sent[q], refsTerms)
                    if t:
                        terms.append([q, t])
                    u = findUnits(sent[q], refsUM)
                    if u:
                        units.append([q, u])
                        values.append([q, findValues(sent[q], u + t)])

            if (units and 1 in [x[0] for x in units] and ([y for x in values for y in x[1]] or terms)) or noLimit:
                    win = win.append({'UM': units,
                                      'NumVal': values,
                                      'Terms': terms,
                                      'Segment': info[i],
                                      'Text': sent,
                                      'Doc': file[i]},
                                     ignore_index=True)
    return win, tokenicer, sentencer


# extract the units in a tokenized sentence
def findUnits(sent, refs):
    res = []
    units = []
    for u in refs:
        w = refs[u][-1]
        for m in matcher.match_sequence(simple_tokens(w), simple_tokens(sent)):
            res.append([w, m[1], m[2], refs[u][0]])
    for t in res:
        valid = True
        for j in res:
            if (t[1] >= j[1] and t[2] <= j[2]) and (t[1:3] != j[1:3]):
                valid = False
        if valid and t not in units:
            units.append(t)
    return units


# extract the numeric values in a tokenized sentence
def findValues(sent, units):
    values = []
    units = [z for y in [range(x[1], x[2]) for x in units] for z in y]
    temp = []
    for i in range(0, len(sent)):  # vérifie que on est pas dans les um
        if i not in units and (
                (len(re.findall('[\d±*×()]', sent[i])) >= (len(sent[i]) / 1.8) or  # token avec une majorité de chiffres
                 re.findall('[^\w!"#$%&\',/:;?@\[\]_`{|}.]', sent[i]))):  # OU sans symboles non math
            temp.append([sent[i], i, i + 1])
    if temp:
        val = [[temp[0][0]], temp[0][1], temp[0][2]]
        i = 1
        while i < len(temp):
            if val[2] == temp[i][1]:
                val = [val[0] + [temp[i][0]], val[1], temp[i][2]]
                i += 1
            else:
                values.append(val)
                val = [[temp[i][0]], temp[i][1], temp[i][2]]
                i += 1
        while val[0][0] in ['(', ')']:
            val[0] = val[0][1:]
            if not val[0]:
                break
        if val[0]:
            while val[0][-1] in ['(', ')']:
                val[0] = val[0][:-1]
                if not val[0]:
                    break
        if val[0]:
            values.append(val)

    copy = values.copy()
    for i in copy:
        if not re.findall('\d', ''.join(i[0])):
            values.remove(i)

    res = []
    for i in values:
        splited = False
        while i[0][0] in ['(', ')']:
            i = [i[0][1:], i[1]+1, i[2]]
        while i[0][-1] in ['(', ')']:
            i = [i[0][:-1], i[1], i[2]-1]
        if re.findall('\d-\d', ''.join(i[0])):
            splited = True
            temp = [[], 0, 0]
            for j in range(0, len(i[0])):
                if i[0][j] != '-':
                    temp = [temp[0]+[i[0][j]], i[1], max([i[1], temp[2]])+1]
                else:
                    res.append(temp)
                    temp = [[], 0, 0]
            res.append(temp)
        if not splited:
            res.append(i)

    return res


# extract the terms in a tokenized sentence
def findTerms(sent, refs):
    terms = []
    temp = []
    for i in refs:
        w = refs[i][-1]
        for m in matcher.match_sequence(simple_tokens(w), simple_tokens(sent)):
            # print(m)
            temp.append([list(m[0]), m[1], m[2], refs[i][0], refs[i][1], refs[i][2], refs[i][3]])
    for t in temp:
        valid = True
        for j in temp:
            if (t[1] >= j[1] and t[2] <= j[2]) and (t[3] == j[3]) and (t[1:3] != j[1:3]):
                valid = False
        if valid and t[0:4] not in [x[0:4] for x in terms]:
            terms.append(t)
    return terms


def simple_tokens(token_list):
    token_list = [x.lower() for x in token_list]
    # token_list = []
    return token_list


def addArgument(df, arg, track, span, disamb, sentence, window, segment, doc):
    df = df.append({'Argument': arg,
                    'Track': track,
                    'Span': span,
                    'Disambiguation': disamb if len(''.join(disamb[0])) > 1 else [track[0].split('_'), None, None],
                    'Sentence': sentence,
                    'Window': window,
                    'Segment': segment,
                    'Document': doc},
                   ignore_index=True)
    return df


def alertOverlap(values, copy, overlap):
    res = []
    position = [(x[1], x[2]) for x in values]
    for i in range(0, len(position)):
        for j in range(0, len(position)):
            if i == j:
                continue
            if min(position[i][1], position[j][1]) - max(position[i][0], position[j][0]) > 0:
                duo = [x for x in values if (x[1], x[2]) in (position[i], position[j])]
                line = str(copy) + '\t' + str(duo) + '\n'
                if duo not in res:
                    res.append(duo)
                    overlap.write(line)
    return res


def transform_sentence(sentence, terms, numval, units, overlap):
    replace = [x + ['cm'] for x in units] + [x + ['10'] for x in numval] + [x + ['concept'] for x in terms]
    replace.sort(key=takeSecond)
    over = alertOverlap(replace, sentence, overlap)

    if over:
        changes = []
        for o in over:
            if changes:
                for c in range(0, len(changes)):
                    if o[0] in changes[c] and o[1] not in changes[c]:
                        changes[c].append(o[1])
                    elif o[1] in changes[c] and o[0] not in changes[c]:
                        changes[c].append(o[0])
                    else:
                        changes.append(o)
            else:
                changes.append(o)
        versions = []
        for v in range(0, max([len(x) for x in changes])):
            keep = [c[v] if v < len(c) else c[0] for c in changes]
            discard = [x for c in changes for x in c if x not in keep]
            versions.append([r for r in replace if r not in discard])
    else:
        versions = [replace]

    res = []
    for v in versions:
        decal = 0
        copy = sentence[:]
        trad = {'cm': [], '10': [], 'concept': []}
        for r in v:
            copy[r[1] - decal:r[2] - decal] = [r[-1]]
            trad[r[-1]].append([r[1], r[2], r[1] - decal])
            decal += r[2] - r[1] - 1
        num = [x for x in numval if (x[1], x[2]) in [(y[0], y[1]) for y in trad['10']]]
        uni = [x for x in units if (x[1], x[2]) in [(y[0], y[1]) for y in trad['cm']]]
        ter = [x for x in terms if (x[1], x[2]) in [(y[0], y[1]) for y in trad['concept']]]
        res.append([copy, trad, num, uni, ter])

    return res


def desambigUM(df, n, origin_unit, sentence, window, seg, doc, data, head_of_num, ter, trad, tolerance, quantityArg,
               added, addimentionnalArg):
    if len(origin_unit[-1]) == 1 and origin_unit[-1][0] not in [x[0] for x in addimentionnalArg]:
        df = addArgument(df, origin_unit[-1][0], [origin_unit[-1][0], 0, 'PrefLabel'],
                         [n, origin_unit[:-1]], [origin_unit[-1][0].split('_'), None, None], sentence, window,
                         seg,
                         doc)
        added = True
    elif len(origin_unit[-1]) >= 2:
        candidat = data[head_of_num.head - 1]
        if candidat.text == 'concept':
            for o in [x for x in trad['concept'] if x[-1] == int(candidat.id) - 1]:
                origin_term = [y for y in ter if y[1:3] == o[0:2]][0]
                if origin_term[3] in origin_unit[-1]:
                    df = addArgument(df, origin_term[3], origin_term[4:], [n, origin_unit[:-1]],
                                     origin_term[0:3], sentence, window, seg, doc)
                    added = True
        else:
            closest_candidates = [(y[1], y[0]) for y in
                                  [((abs(z[0] + 1), z[1]) if z[0] < 0 else (z[0], z[1])) for z in
                                   [(int(x.id) - int(candidat.id), int(x.id)) for x in data if
                                    x.text == 'concept']] if y[0] < tolerance]
            closest_candidates.sort(key=takeSecond)
            for candi in closest_candidates:
                for o in [x for x in trad['concept'] if x[-1] == int(data[candi[0] - 1].id) - 1]:
                    origin_term = [y for y in ter if y[1:3] == o[0:2]][0]
                    if origin_term[3] in origin_unit[-1]:
                        df = addArgument(df, origin_term[3], origin_term[4:], [n, origin_unit[:-1]],
                                         origin_term[0:3], sentence, window, seg, doc)
                        added = True
                        # break
    else:
        candidat = data[head_of_num.head - 1]
        if candidat.text == 'concept':
            origin_term = [y for y in ter if
                           y[1:3] == [x for x in trad['concept'] if x[-1] == int(candidat.id) - 1][0][
                                     0:2]][0]
            if origin_term[3] in quantityArg:
                df = addArgument(df, origin_term[3], origin_term[4:], [n, origin_unit[:-1]],
                                 origin_term[0:3], sentence, window, seg, doc)
                added = True
        else:
            closest_candidates = [(y[1], y[0]) for y in
                                  [((abs(z[0] + 1), z[1]) if z[0] < 0 else (z[0], z[1])) for z in
                                   [(int(x.id) - int(candidat.id), int(x.id)) for x in data if
                                    x.text == 'concept']] if y[0] < tolerance]
            closest_candidates.sort(key=takeSecond)
            for candi in closest_candidates:
                for o in [x for x in trad['concept'] if x[-1] == int(data[candi[0] - 1].id) - 1]:
                    origin_term = [y for y in ter if y[1:3] == o[0:2]][0]
                    if origin_term[3] in quantityArg:
                        df = addArgument(df, origin_term[3], origin_term[4:], [n, origin_unit[:-1]],
                                         origin_term[0:3], sentence, window, seg, doc)
                        added = True
    # if not added and len(origin_unit[-1]) >= 2:
    #     for o in origin_unit[-1]:
    #         df = addArgument(df, o, [o, 0, 'PrefLabel'], [n, origin_unit[:-1]], [o.split('_'), None, None],
    #                          sentence, window, seg, doc)
    return df, added


def buildArgument(win, tolerance):
    nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='en', package='partut',
                          tokenize_pretokenized=True)
    df = pd.DataFrame(None, columns=['Argument', 'Track', 'Span', 'Disambiguation', 'Sentence', 'Window', 'Segment',
                                     'Document'])

    w_numval = win.NumVal.values
    w_units = win.UM.values
    w_terms = win.Terms.values
    w_win = win.Text.values
    w_seg = win.Segment.values
    w_doc = win.Doc.values

    quantityArg = []
    addimentionnalArg = []
    symbolicArg = []
    desambig = pd.read_csv(r'work_files/desambig.csv')
    for i in pd.read_csv('work_files/naryrelations.csv').itertuples():
        if i.Type == 'QUANTITY':
            if 'one' in desambig[desambig.Concept == i.Argument].Unit.values:
                addimentionnalArg.append(i.Argument)
            else:
                quantityArg.append(i.Argument)
        elif i.Type == 'SYMBOLIC':
            symbolicArg.append(i.Argument)
    quantityArg = list(set(quantityArg))
    # addimentionnalArg = list(set(addimentionnalArg))  # besoin d'ajouter le symbolic-arg associé
    addimentionnalArg = [('component_qty_value', 'impact_factor_component')]
    symbolicArg = list(set(symbolicArg))

    overlap = open(r'work_files/OVERLAP.txt', 'w', encoding='utf-8')

    for w in tqdm(range(0, len(win))):
        numval_r = ast.literal_eval(w_numval[w])
        units_r = ast.literal_eval(w_units[w])
        terms_r = ast.literal_eval(w_terms[w])
        window_r = ast.literal_eval(w_win[w])
        seg_r = ast.literal_eval(w_seg[w])
        doc_r = w_doc[w]

        if not numval_r and not units_r and not terms_r:
            continue

        for i in range(0, len(window_r)):
            sentence = window_r[i]
            if not sentence:
                continue
            terms = [x for x in terms_r if x[0] == i][0][1] if [x for x in terms_r if x[0] == i] else []
            numval = [x for x in numval_r if x[0] == i][0][1] if [x for x in numval_r if x[0] == i] else []
            units = [x for x in units_r if x[0] == i][0][1] if [x for x in units_r if x[0] == i] else []
            for term in terms:
                if term[3] in symbolicArg:
                    df = addArgument(df, term[3], term[4:], term[0:3], term[0:3], sentence, window_r, seg_r, doc_r)

            if not numval_r and not units_r:
                continue
            copy = list(sentence)
            versions = transform_sentence(copy, terms, numval, units, overlap)  # [(version_1, traduction_1), ...]
            for v in versions:
                copy, trad, num, uni, ter = v[0], v[1], v[2], v[3], v[4]
                data = nlp(' '.join(copy)).sentences[0].words
                for n in [x for x in num]:
                    added = False
                    converted_num = data[[x for x in trad['10'] if x[0:2] == n[1:3]][0][-1]]
                    head_of_num = data[converted_num.head - 1]

                    if head_of_num.text == 'cm':  # numval is dependent to unit
                        origin_unit = [y for y in uni if
                                       y[1:3] == [x for x in trad['cm'] if x[-1] == int(head_of_num.id) - 1][0][0:2]][0]
                        df, added = desambigUM(df, n, origin_unit, sentence, window_r, seg_r, doc_r, data, head_of_num,
                                               ter, trad, tolerance, quantityArg, added, addimentionnalArg)

                    elif head_of_num.text == 'concept':  # numval is dependent to concept
                        # print(head_of_num)
                        for o in [x for x in trad['concept'] if x[-1] == int(head_of_num.id) - 1]:
                            origin_term = [y for y in ter if y[1:3] == o[0:2]][0]
                            argument = [x for x in addimentionnalArg if x[1] == origin_term[3]]
                            if argument:
                                df = addArgument(df, argument[0][0], [argument[0][0], 0, 'PrefLabel'],
                                                 [n, origin_term[0:3]],
                                                 origin_term[0:3], sentence, window_r, seg_r, doc_r)
                                added = True

                    if not added:  # try with unit close to numval
                        closest_candidates = [(y[1], y[0]) for y in
                                              [((abs(z[0] + 1), z[1]) if z[0] < 0 else (z[0], z[1])) for z in
                                               [(int(x.id) - int(data[int(converted_num.id) - 1].id), int(x.id)) for x
                                                in
                                                data if
                                                x.text == 'cm']] if y[0] < tolerance / 2]
                        closest_candidates.sort(key=takeSecond)
                        for candi in closest_candidates:
                            # if added:
                            #     break
                            for o in [x for x in trad['cm'] if x[-1] == int(data[candi[0] - 1].id) - 1]:
                                origin_unit = [y for y in uni if y[1:3] == o[0:2]][0]
                                df, added = desambigUM(df, n, origin_unit, sentence, window_r, seg_r, doc_r, data,
                                                       head_of_num, ter, trad, tolerance, quantityArg, added,
                                                       addimentionnalArg)

                    if not added:  # try with close concepts
                        closest_candidates = [(y[1], y[0]) for y in
                                              [((abs(z[0] + 1), z[1]) if z[0] < 0 else (z[0], z[1])) for z in
                                               [(int(x.id) - int(data[int(converted_num.id) - 1].id), int(x.id)) for x
                                                in
                                                data if x.text == 'concept']] if y[0] < tolerance]
                        closest_candidates.sort(key=takeSecond)
                        for candidat in closest_candidates:
                            for o in [x for x in trad['concept'] if x[-1] == candidat[0] - 1]:
                                origin_term = [y for y in ter if y[1:3] == o[0:2]][0]
                                argument = [x for x in addimentionnalArg if x[1] == origin_term[3]]
                                if argument:
                                    df = addArgument(df, argument[0][0], [argument[0][0], 0, 'PrefLabel'], [n],
                                                     origin_term[0:3], sentence, window_r, seg_r, doc_r)
                                    added = True
                                    # break
                            # if added:
                            #     break

    overlap.close()
    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]
    return df


def takeSecond(elem):
    return elem[1]


def cleanDF(df):
    df.dropna(subset=['Argument'], inplace=True)

    quantityArg = []
    symbolicArg = []
    addimentionnalArg = []
    desambig = pd.read_csv(r'work_files/desambig.csv')
    for i in pd.read_csv('work_files/naryrelations.csv').itertuples():
        if i.Type == 'QUANTITY':
            if 'one' in desambig[desambig.Concept == i.Argument].Unit.values:
                addimentionnalArg.append(i.Argument)
            else:
                quantityArg.append(i.Argument)
        elif i.Type == 'SYMBOLIC':
            symbolicArg.append(i.Argument)
    quantityArg = list(set(quantityArg))
    # addimentionnalArg = list(set(addimentionnalArg))  # besoin d'ajouter le symbolic-arg associé
    addimentionnalArg = [('component_qty_value', 'impact_factor_component')]
    symbolicArg = list(set(symbolicArg))
    df['Type'] = ['QUANTITY' if x in quantityArg else
                  'SYMBOLIC' if x in symbolicArg else
                  'ADDIMENTIONNAL'
                  for x in df['Argument']]

    df = df.rename(columns={'Argument': 'Argument',
                            'Track': 'Node',
                            'Type': 'Type',
                            'Span': 'Original_value',
                            'Disambiguation': 'Attached_value',
                            'Segment': 'Segment',
                            'Sentence': 'Sentence',
                            'Window': 'Window',
                            'Document': 'Document',

                            'DC_Tree': 'DC_Tree',

                            'TF_segment_term_top': 'TF_segment_term_top',
                            'ICF_segment_term_top': 'ICF_segment_term_top',
                            'TF_segment_arg_top': 'TF_segment_arg_top',
                            'ICF_segment_arg_top': 'ICF_segment_arg_top',

                            'TF_segment_term_bot': 'TF_segment_term_bot',
                            'ICF_segment_term_bot': 'ICF_segment_term_bot',
                            'TF_segment_arg_bot': 'TF_segment_arg_bot',
                            'ICF_segment_arg_bot': 'ICF_segment_arg_bot',

                            'TF_classic_term': 'TF_classic_term',
                            'IDF_classic_term': 'IDF_classic_term'
                            })

    df = df.drop_duplicates(['Argument', 'Node', 'Original_value', 'Attached_value', 'Sentence', 'Document'],
                            keep='first')

    columns_to_normalize = ['DC_Tree',
                            'TF_segment_term_top', 'ICF_segment_term_top', 'TF_segment_arg_top', 'ICF_segment_arg_top',
                            'TF_segment_term_bot', 'ICF_segment_term_bot', 'TF_segment_arg_bot', 'ICF_segment_arg_bot',
                            'TF_classic_term', 'IDF_classic_term']
    for c in columns_to_normalize:
        df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    if 'corp.csv' not in os.listdir('work_files/'):
        corp = corpusCrawler.parse_all('corpus')
        # corp = corp[corp.Tag != 'info']
        # corp = corp[~corp['Attribute'].isin(['keywords', 'outline', 'acknowledgements', 'references'])]           # optional
        corp.to_csv('work_files/corp.csv', encoding='utf-8')  # Document|Tag|Attribut|Texte
    corp = pd.read_csv(r'work_files/corp.csv')

    print("#####\n#ONTO")
    voca = ontoCrawler.importOnto(['permeabilities', 'impact_factor_component_relation'])
    print('#voca manuel')
    voca = addManuVoc(voca)
    print('#fastr')
    voca = fastring(voca)
    print('#acrotr')
    voca = AcrOtr.addAcroFind(voca, corp, fastr=True, threshold=0.7)
    voca.to_csv(r'work_files/vocaExtended.csv', encoding='utf-8')

    print('VOCABULARY CREATED\n#####\n#####UNITS')
    um = ontoCrawler.getAllUnits()
    um.to_csv('work_files/unitsRTO.csv', encoding='utf-8')
    umF = umFinder.find_all_units(corp, list(um.PrefLabel.values) + list(um.AltLabel.values))
    umF.to_csv('work_files/unitsVar.csv', encoding='utf-8')
    print('#desambiguisation')
    desambig = ontoCrawler.getAssociation()
    desambig.to_csv('work_files/desambig.csv', encoding='utf-8')
    print('#unit file')
    unit = createUnitFile(um, umF, desambig)
    unit.to_csv('work_files/units.csv', encoding='utf-8')

    print('#####\n# BEGIN EXTRACTION\n#####\n# look for Windows')
    if 'win.csv' not in os.listdir('work_files/'):
        win, tokenicer, sentencer = extractWindow(corp, unit, voca, debut=-1, fin=1, noLimit=True)
        win.to_csv('work_files/win.csv', encoding='utf-8')
    else:
        print('EXIST')
    win = pd.read_csv('work_files/win.csv')

    print('# extract Arguments')
    if 'argCLEAN.csv' not in os.listdir('work_files/'):
        # arg = pd.DataFrame(columns=['Argument', 'Track', 'Span', 'Disambiguation', 'Sentence', 'Window', 'Segment',
        #                             'Document'])
        # sample = 2500
        # iteration = 2
        # while iteration*sample < len(win):
        #     print('SAMPLE '+str(iteration+1))
        #     arg.append(buildArgument(win[iteration*sample:(iteration+1)*sample], tolerance=20))
        #     iteration += 1
        arg = buildArgument(win, tolerance=20)
        arg.to_csv('work_files/arg.csv', encoding='utf-8')
        arg = pd.read_csv('work_files/arg.csv')
        for col in arg.columns:
            if 'Unnamed' in col:
                del arg[col]
        arg.drop_duplicates(['Argument', 'Track', 'Span', 'Disambiguation', 'Sentence', 'Segment', 'Document'],
                            keep='first').to_csv('work_files/argCLEAN.csv', encoding='utf-8')
    else:
        print('EXIST')
    arg = pd.read_csv('work_files/argCLEAN.csv')

    print('# add Tf-Icf to ' + str(len(arg)) + ' results')
    # if 'segment_context_Scores.csv' not in os.listdir('work_files/'):
    #     print('context scores', end='')
    #     SegmentAnalyse.get_contextScores(arg)
    # if 'classic_Scores.csv' not in os.listdir('work_files/'):
    #     print('classic scores', end='')
    #     SegmentAnalyse.get_classic_scores(arg)

    SegmentAnalyse.prepare_Scores(arg, corp)
    res = SegmentAnalyse.add_Scores(arg)

    print('\n ########################################## \n FINALISATION \n ##########################################')
    res.to_csv('resPHASE1/res.csv', encoding='utf-8')
    res = cleanDF(res)
    res.to_csv('resPHASE1/resFINAL.csv', encoding='utf-8')


if __name__ == '__main__':
    main()


"""
REVOIR LA RECONNAISSANCE DES ARG ADDIMENTIONNELS     
"""
