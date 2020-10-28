# coding: utf-8
from __future__ import unicode_literals

"""
Different methods to process the ontologies
"""

import re
import ast
import pandas as pd
from rdflib import *


def clean(txt):
    txt = re.sub('−', '-', txt)
    txt = re.sub('_', ' ', txt)
    txt = re.sub('·', ' ', txt)
    txt = re.sub('/', ' / ', txt)
    txt = re.sub('%', ' % ', txt)
    txt = re.sub('\n+', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub("′", "'", txt)
    txt = re.sub('\xa0', ' ', txt)
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    # txt[i] = re.sub('', ' ', txt[i])
    txt = re.sub(' +', ' ', txt)
    while txt[0] == ' ':
        txt = txt[1:]
    while txt[-1] == ' ':
        txt = txt[:-1]
    return txt


# return the association between quantitative arguments and units of measure
def getAssociation():
    newUM = pd.read_csv('work_files/unitsVar.csv', encoding='utf-8')
    onto = open('input_files/TRANSMAT.owl', encoding='utf-8').read()
    UM = pd.read_csv('work_files/unitsRTO.csv')
    df = pd.DataFrame(None, columns=['Concept', 'Unit'])
    cons = list(set(pd.read_csv('work_files/naryrelations.csv').Argument.values.tolist()))
    for c in cons:
        c2 = c.split('#')[-1]
        num = len(re.findall(' ', re.findall('\n +<skos:Concept rdf:ID="' + c2, onto)[0])) - 1
        area = re.findall(
            ' {' + str(num) + '}<skos:Concept rdf:ID="' + c2 + '">.+?\n {' + str(num) + '}</skos:Concept>', onto,
            re.DOTALL)
        if area:
            area = re.sub('<core:hasOperand.+?</core:hasOperand.+?\n', '', area[0], flags=re.DOTALL)  # A VERRIFIER
            units = list(set([re.findall('".+?"', u)[0][1:-1].replace('#', '').replace('_', ' ').lower()
                              for u in re.findall('<core:.*?Unit.+', area) if re.findall('".+?"', u)]))
            units += list(set([re.findall('".+?"', u)[0][1:-1].replace('#', '').replace('_', ' ').lower()
                               for u in re.findall('<skos:Concept rdf:about=.+', area) if re.findall('".+?"', u)]))
            units += list(set(UM[UM.PrefLabel.map(lambda x: x.lower() if isinstance(x, str) else x).isin(units)]
                              .AltLabel.values.tolist()))
        else:
            units = ['string'] if re.findall('<owl:Class rdf:ID="' + c2 + '"/>', onto) else ''
        for u in units:
            df = df.append({'Concept': c, 'Unit': u}, ignore_index=True)
    for tupl in df.itertuples():
        for j in newUM[newUM.UMrto == tupl.Unit].itertuples():
            df = df.append({'Concept': tupl.Concept, 'Unit': j.UMnew}, ignore_index=True)
    return df


# extract and order the units from an ontology
def getAllUnits():
    g = Graph()
    g.parse('input_files/global_unit.owl')

    qres = g.query(
        """
        SELECT ?category ?prefLabel ?altLabel
        WHERE {
            ?x rdf:type ?category .
            ?x skos:prefLabel ?prefLabel .
            ?x skos:altLabel ?altLabel .
        }""")

    typee = []
    pref = []
    alt = []

    for row in qres:
        if row[1].language == 'en' and row[2].language == 'en':
            if re.findall('\D', clean(row[2])):
                typee.append(re.findall('#.+', row[0])[0][1:])
                pref.append(clean(row[1]))
                if re.findall('\D', clean(row[2])):
                    alt.append(clean(row[2]))
                else:
                    alt.append('')

    df = pd.DataFrame({'Type': typee, 'PrefLabel': pref, 'AltLabel': alt})
    # print("### DONE")
    return df


# return the distance between the considered node and its original argument
def depth(g, df, father, grandfather, d):
    d += 1
    argument_child = g.query("SELECT ?x WHERE {"
                             "?x rdfs:subClassOf ?arg"
                             "}",
                             initBindings={'arg': URIRef(father)})
    for a in argument_child:
        a = a[0]
        pref = []
        alt = []
        for l in labelA(a, g):
            if l[0].language == 'en' and l[0].value not in pref:
                pref.append(clean(l[0].value))
            if l[1] and l[1].language == 'en' and l[1].value not in alt:
                alt.append(clean(l[1].value))
        df = df.append({'Argument': grandfather.split('#')[-1],
                        'Node': a.split('#')[-1],
                        'Depth': d,
                        'PrefLabel': pref,
                        'AltLabel': alt}, ignore_index=True)
        df = depth(g, df, a, grandfather, d)
    return df


# return the labels of a node
def labelA(a, g):
    return g.query("SELECT DISTINCT ?prefLabel ?altLabel WHERE {"
                   "?arg skos:prefLabel ?prefLabel "
                   "OPTIONAL {?arg skos:altLabel ?altLabel}"
                   "}",
                   initBindings={'arg': URIRef(a)})


# def onePermeability(voca, relation):
#
#     voca2 = pd.DataFrame(['Argument', 'Node', 'Depth', 'PrefLabel', 'AltLabel'])
#     voca2 = voca2.append({'Argument': ['permeability'], 'Node': ['permeability'], 'Depth': [0],
#                           'PrefLabel': ['permeability'], 'AltLabel': ['']}, ignore_index=True)
#     for v in voca.itertuples():
#         if v.Argument in ['co2_permeability', 'o2_permeability', 'h2o_permeability']:
#             voca2 = voca2.append({'Argument': 'permeability', 'Node': v.Node, 'Depth': v.Depth + 1,
#                                   'PrefLabel': v.PrefLabel, 'AltLabel': v.AltLabel},
#                                  ignore_index=True)
#         else:
#             voca2 = voca2.append({'Argument': v.Argument, 'Node': v.Node, 'Depth': v.Depth,
#                                   'PrefLabel': v.PrefLabel, 'AltLabel': v.AltLabel},
#                                  ignore_index=True)
#
#     data = {'Relation': ['permeability_relation', 'permeability_relation', 'permeability_relation',
#                          'permeability_relation', 'permeability_relation', 'permeability_relation',
#                          'permeability_relation', 'permeability_relation', 'impact_factor_component_relation',
#                          'impact_factor_component_relation', 'impact_factor_component_relation',
#                          'impact_factor_component_relation'],
#             'Argument': ['thickness', 'permeability', 'temperature', 'partial_pressure', 'method', 'packaging',
#                          'partial_pressure_difference', 'relative_humidity', 'component_qty_value',
#                          'impact_factor_component', 'packaging', 'packnumber'],
#             'Type': ['QUANTITY', 'QUANTITY', 'QUANTITY', 'QUANTITY', 'SYMBOLIC', 'SYMBOLIC', 'QUANTITY', 'QUANTITY',
#                      'QUANTITY', 'SYMBOLIC', 'SYMBOLIC', 'QUANTITY']}
#     relation2 = pd.DataFrame(data)
#
#     return voca2, relation2


# process the ontology to extract arguments and labels of the given relations
def importOnto(naries):
    g = Graph()
    g.parse('input_files/TRANSMAT.owl')
    onto = open('input_files/TRANSMAT.owl', encoding='utf-8').read()

    df = pd.DataFrame(None, columns=['Argument', 'Node', 'Depth', 'PrefLabel', 'AltLabel'])
    df2 = pd.DataFrame(None, columns=['Relation', 'Argument', 'Type'])

    for nary in naries:
        relations_arguments = g.query("SELECT DISTINCT ?relation ?argument WHERE {"
                                      "?relation rdfs:subClassOf ?perm ."
                                      "?relation ?child ?noeud ."
                                      "{?noeud ?primar ?argument} UNION {?noeud ?second ?argument}"
                                      "}",
                                      initBindings={
                                          'perm': URIRef('http://opendata.inra.fr/resources/hSC9z#' + nary),
                                          'primar': URIRef('http://www.w3.org/2002/07/owl#allValuesFrom'),
                                          'second': URIRef('http://www.w3.org/2002/07/owl#someValuesFrom')})

        if len(relations_arguments) == 0:
            relations_arguments = g.query("SELECT DISTINCT ?perm ?argument WHERE {"
                                          "?perm ?child ?noeud ."
                                          "{?noeud ?primar ?argument} UNION {?noeud ?second ?argument}"
                                          "}",
                                          initBindings={
                                              'perm': URIRef('http://opendata.inra.fr/resources/hSC9z#' + nary),
                                              'primar': URIRef('http://www.w3.org/2002/07/owl#allValuesFrom'),
                                              'second': URIRef('http://www.w3.org/2002/07/owl#someValuesFrom')})
        relations = []
        rel_list = []
        arguments = []
        typee = []
        arg_list = []
        for ra in relations_arguments:
            relations.append(ra[0].split('#')[-1])
            arguments.append(ra[1].split('#')[-1])
            spaces = len(re.findall('\n +?<skos:Concept rdf:ID="' + ra[1].split('#')[-1] + '">', onto)[0].split('<')[0][
                         1:]) if re.findall('\n +?<skos:Concept rdf:ID="' + ra[1].split('#')[-1] + '">', onto) else 0
            area = re.findall(' ' * spaces + '<skos:Concept rdf:ID="' + ra[1].split('#')[
                -1] + '">.+?' + '\n' + ' ' * spaces + '</skos:Concept>\n', onto, re.DOTALL)
            if area:
                if 'hasNumericalValue' in ' '.join(re.findall('<owl:onProperty rdf:resource=".+?/>\n', area[0])):
                    typee.append('QUANTITY')
                else:
                    typee.append('SYMBOLIC')
            else:
                typee.append('SYMBOLIC')
            if ra[0] not in rel_list:
                rel_list.append(ra[0])
            if ra[1] not in arg_list:
                arg_list.append(ra[1])
            df2 = df2.append({'Relation': ra[0].split('#')[-1], 'Argument': ra[1].split('#')[-1], 'Type': typee[-1]},
                             ignore_index=True)

        for a in arg_list:
            d = 0
            pref = []
            alt = []
            for l in labelA(a, g):
                if l[0].language == 'en' and l[0].value not in pref:
                    pref.append(clean(l[0].value))
                if l[1] and l[1].language == 'en' and l[1].value not in alt:
                    alt.append(clean(l[1].value))
            df = df.append({'Argument': a.split('#')[-1],
                            'Node': a.split('#')[-1],
                            'Depth': d,
                            'PrefLabel': pref,
                            'AltLabel': alt}, ignore_index=True)
            df = depth(g, df, a, a, d)

    # df, df2 = onePermeability(df, df2)

    df.to_csv('work_files/vocOnto.csv', encoding='utf-8')
    df2.to_csv('work_files/naryrelations.csv', encoding='utf-8')

    return df


def main():
    print('ontoCrawler')


if __name__ == '__main__':
    main()
