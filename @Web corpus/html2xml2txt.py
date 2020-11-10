# coding: utf-8

import re
import os
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from html_table_extractor.extractor import Extractor


def clean_html(html):
    for r in re.findall('<sup>.+?</sup>', html, re.DOTALL):
        html = re.sub(re.escape(r), '^' + r[5:-6], html)
    for r in re.findall('<math>.+?</math>', html, re.DOTALL):
        html = re.sub(re.escape(r), '', html)
    html = re.sub(' +', ' ', html)
    return html


def clean_4_xml(txt):
    corresponding = [['&amp;', '&'], ['&lt;', '<'], ['&gt;', '>'], ['&quot;', '"'], ['&apos;', "'"]]
    for i in range(0, len(corresponding)):
        txt = txt.replace(corresponding[i][1], corresponding[i][0])
    txt = re.sub('%', ' % ', txt)
    txt = re.sub(' +', ' ', txt)
    return txt


def get_figures(figures):
    res = []
    for f in figures:
        labels = f.find_all('span', attrs={'class': 'label'})
        captions = f.find_all('span', attrs={'class': 'captions'})
        contents = f.find_all('img')
        fig = {'label': labels[0].text if labels else '',
               'caption': captions[0].text.replace(labels[0].text, '') if captions else '',
               'content': contents[0]['src'] if contents else ''
               }
        if fig['label']:
            res.append(fig)
    return res


def get_tables(tables):
    res = []
    for t in tables:
        labels = t.find_all('span', attrs={'class': 'label'})
        captions = t.find_all('span', attrs={'class': 'captions'})
        legend = t.find_all('p', attrs={'class': 'legend'})
        footnotes = t.find_all('dl', attrs={'class': 'footnotes'})
        for t2 in t.find_all('table'):
            extractor = Extractor(t2)
            extractor.parse()
            content = extractor.return_list()
            tab = {'label': labels[0].text if labels else '',
                   'caption': captions[0].text.replace(labels[0].text, '') if captions else '',
                   'legend': [x.text for x in legend] if legend else '',
                   'footnote': [x.text for x in footnotes] if footnotes else '',
                   'content': content
                   }
            res.append(tab)
    return res


def get_sections(soup, level, file):
    res = []
    for s in soup:
        if s.find('h' + str(level)) and re.findall('\d', s.find('h' + str(level)).text):
            subsection = [x.extract() for x in s.find_all('section', recursive=False)]
            figs = [x.extract() for x in s.find_all('figure')]
            tabs = [x.extract() for x in s.find_all('div', attrs={'class': 'tables'})]
            section = {'level': level - 2,
                       'section': (s.find('h' + str(level)).text.split(' ')[0],
                                   ' '.join(s.find('h' + str(level)).text.split(' ')[1:])),
                       'text': [x.text for x in s.find_all('p')],
                       'childs': get_sections(subsection, level=level + 1, file=file),
                       'figures': get_figures(figs),
                       'tables': get_tables(tabs)
                       }
            res.append(section)
    return res


def build_pretty_table(xml, t):  # pas de repetition dans les cases
    table = t['content']
    for x in range(0, len(table)):
        xml = xml + 'Line:'
        for y in range(0, len(table[x])):
            xml = xml + clean_4_xml('\t' + table[x][y] + '\t' * (
                        (max([len(table[l][y]) for l in range(0, len(table))]) // 4) - (
                            len(table[x][y]) // 4)) + '\t|')
        xml = xml[:-1] + '.\n'
    return xml


def add_sections(section, xml):
    xml = xml + '<' + section['level'] * 'sub' + 'part level="' + clean_4_xml(
        section['section'][0]) + '" type="' + clean_4_xml(
        section['section'][-1]) + '" name="' + clean_4_xml(section['section'][-1]) + '">\n'
    xml = xml + clean_4_xml('\n'.join(section['text']) + '\n' if section['text'] else '')
    for f in section['figures']:
        xml = xml + '<fig level="' + clean_4_xml(f['label']).split(' ')[
            -1] + '" type="figure" name="' + clean_4_xml(f['label']) + '">\n'
        xml = xml + '<caption>' + ' '.join(
            clean_4_xml(f['caption']).split('.')[1:]) + '.</caption>\n'
        xml = xml + '<content>' + clean_4_xml(f['content']) + '</content>\n'
        xml = xml + '</fig>\n'
    for t in section['tables']:
        xml = xml + '<fig level="' + clean_4_xml(t['label']).split(' ')[
            -1] + '" type="table" name="' + clean_4_xml(t['label']) + '">\n'
        xml = xml + '<caption>' + ' '.join(
            clean_4_xml(t['caption']).split('.')[1:]) + '.</caption>\n'
        xml = xml + '<legend>' + ' '.join(
            clean_4_xml('\n'.join(t['legend']) if t['legend'] else '').split('.')[1:]) + (
                  '.' if t['legend'] else '') + '</legend>\n'
        xml = xml + '<footnote>' + ' '.join(
            clean_4_xml('\n'.join(t['legend']) if t['footnote'] else '').split('.')[1:]) + (
                  '.' if t['footnote'] else '') + '</footnote>\n'
        xml = xml + '<content>\n'
        xml = build_pretty_table(xml, t)
        xml = xml + '\n</content>\n'
        xml = xml + '</fig>\n'
    for c in section['childs']:
        xml = add_sections(c, xml)
    xml = xml + '</' + section['level'] * 'sub' + 'part>\n'
    return xml


def html2xml(file):
    html = open('html/' + file, 'r', encoding='utf-8').read()
    html = clean_html(html)
    soup = BeautifulSoup(html, 'html.parser')
    soup = soup.article
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<doc>\n'

    journal = list(set([x.text for x in soup.find('div', attrs={'class': 'Publication',
                                                                'id': 'publication'}).recursiveChildGenerator()
                        if x.name is not None]))
    xml = xml + '<info type="publication">' + clean_4_xml('\n'.join(journal)) + '\n</info>\n'

    doi = soup.find('a', attrs={'class': 'doi'}).text
    xml = xml + '<info type="DOI">\n' + clean_4_xml(doi) + '\n</info>\n'

    authors = [x.text for x in
               soup.find_all('span', attrs={'class': ['text surname', 'text given-name']})]
    xml = xml + '<info type="authors">\n' + clean_4_xml('\n'.join(authors)) + '\n</info>\n'

    title = soup.find('span', attrs={'class': 'title-text'}).text
    xml = xml + '<info type="title" name="title">\n' + clean_4_xml(title) + '\n</info>\n'

    highlight = [x.text for x in soup.find('div', attrs={
        'class': 'abstract author-highlights'}).recursiveChildGenerator() if
                 x.name == 'dd'] if soup.find('div',
                                              attrs={'class': 'abstract author-highlights'}) else ''
    xml = xml + '<part type="highlight" name="highlight">\n' + clean_4_xml(
        '\n'.join(highlight)) + '</part>\n'

    abstract = [x.text for x in
                soup.find('div', attrs={'class': 'abstract author'}).recursiveChildGenerator() if
                x.name == 'p']
    xml = xml + '<part type="abstract" name ="abstract">\n' + clean_4_xml('\n'.join(abstract))

    keywords = [x.text for x in soup.find_all('div', attrs={'class': 'keyword'})]
    xml = xml + '\n<subpart type="keywords" name="keywords">\n' + '\n'.join(
        ['<key>' + clean_4_xml(x) + '. </key>' for x in keywords]) + '\n</subpart>\n</part>\n'

    content = get_sections(soup.find_all('section'), 2, file)
    with open(r'json/' + ''.join(file.split('.')[:-1]) + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(content, indent=4))
    for c in content:
        xml = add_sections(c, xml)

    xml += '</doc>'
    with open(r'xml/' + ''.join(file.split('.')[:-1]) + '.xml', 'w', encoding='utf-8') as f:
        f.write(xml)


def buildTXT(xml, txt):
    for child in xml:
        if child.attrib and child.attrib[[i for i in child.attrib.keys()][0]] not in ['publication',
                                                                                      'DOI',
                                                                                      'authors']:
            igno = []
            for i in child:
                if i.tag not in ['key', 'caption', 'legend', 'footnote', 'content']:
                    for j in i.itertext():
                        igno.append(j)
            corps = '\n'.join([re.sub('\n', ' ', x) for x in child.itertext() if x not in igno])
            corps = re.sub(' +', ' ', corps)
            if len(corps) > 1:
                while corps[-1] == ' ':
                    corps = corps[:-1]
            txt += '\n ' + ' '.join([k.upper() for k in [child.attrib[j] for j in
                                                         [i for i in child.attrib.keys() if
                                                          i in ['name']]]]) + ' . \n' + corps
            if len(corps) > 5:
                txt += '\n' if corps[-1] == '.' else '.\n'
            txt = buildTXT(child, txt)
    return txt


def main():
    for file in tqdm(os.listdir('html/')):
        html2xml(file)
    for file in tqdm(os.listdir('xml')):
        tree = ET.parse('xml/' + file)
        root = tree.getroot()
        txt = buildTXT(root, '')
        open('cleanTXT/' + file[:-4] + '.txt', 'w', encoding="utf8").write(txt)


if __name__ == '__main__':
    main()
