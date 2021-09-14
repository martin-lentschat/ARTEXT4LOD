# coding: utf-8
from __future__ import unicode_literals

import numpy as np
import warnings


def match_sequence(seq, data_set):
    """
    Return match sequence start,end positions in a dataset
    Parameters
    ----------
    seq : list
        sequence
    data_set : list
        dataset

    """
    n = len(seq)
    if n < 1:
        raise ValueError("Sequence is empty !")

    if isinstance(data_set, list):
        data_set = np.asarray(data_set)
    if isinstance(seq, list):
        seq = np.asarray(seq)
    prefix_ind = np.where(data_set == seq[0])[0]
    results = []
    for idx in prefix_ind:
        start, end = idx, idx + n
        if data_set[start:end].tolist() == seq.tolist():
            # results.append([seq, start, end])
            results.append([data_set[start:end], start, end])
    return results


def match_sequences(seqs, data_set):
    """
    Return match sequence start,end positions in a dataset

    Parameters
    ----------
    seq : list
        sequence
    data_set : list
        dataset

    """
    n = len(seqs)
    if n < 1:
        warnings.warn("Sequence Empty")
        return []

    if isinstance(data_set, list):
        data_set = np.asarray(data_set)
    if isinstance(seqs, list):
        seqs = np.array(seqs)

    prefixes_dict = {seq[0]: (seq, i, len(seq)) for i, seq in enumerate(seqs)}
    prefixes = list(prefixes_dict.keys())
    prefix_ind = np.where(np.isin(data_set, prefixes))[0]
    results = []
    for idx in prefix_ind:
        start, end = idx, idx + prefixes_dict[data_set[idx]][-1]
        if data_set[start:end].tolist() == prefixes_dict[data_set[idx]][0]:
            results.append([prefixes_dict[data_set[idx]][1], start, end])
    return results


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == "__main__":
    print('matcher')

    # import spacy
    # from spacy.matcher import PhraseMatcher
    # import timeit
    #
    #
    # def match_syntagm_text_spacy(text, matcher):
    #     return matcher(nlp(text))
    #
    #
    # def match_syntagm_text_blob(syntagm, text):
    #     from textblob import TextBlob
    #     from textblob_fr import PatternTagger, PatternAnalyzer
    #     blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    #     return match_sequence(syntagm, list(blob.tokenize()))
    #
    #
    # def match_syntagm_text_blob_multi(syntagms, text):
    #     from textblob import TextBlob
    #     from textblob_fr import PatternTagger, PatternAnalyzer
    #     blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    #     return match_sequences(syntagms, list(blob.tokenize()))
    #
    #
    # nlp = spacy.load("fr")
    # matcher = PhraseMatcher(nlp.vocab)
    # matcher.add(1, None, nlp("Donald Trump"))
    # text = "Quel sera le sort des centaines d’anciens djihadistes européens partis combattre en Syrie au cours des" \
    #        "dernières années ? Alors que l’organisation Etat islamique (EI) perd de plus en plus de terrain, le" \
    #        "président américain, Donald Trump, a exhorté, dimanche 17 février, les Européens à rapatrier leurs" \
    #        " ressortissants, retenus en Syrie après avoir rallié le groupe islamiste. « Il n’y a pas d’alternative, car" \
    #        " nous serions forcés de les
    #     libérer », a mis en garde le président américain, s’adressant particulièrement à la Grande-Bretagne, la France,
    #      et l’Allemagne.\nDes représentants des autorités du nord-est de la Syrie qui détiennent ces djihadistes
    #       étrangers ont, pour leur part, nuancé la portée des déclarations du président des Etats-Unis. « Nous ne les
    #        relâcherons pas. Jamais nous ne pourrions faire cela », a affirmé le coresponsable des relations
    #         internationales dans la région, M. Abdulkarim Omar. Ce dernier a toutefois averti les gouvernements
    #          européens que ces djihadistes constituaient des « bombes à retardement ». Il a exhorté leurs pays
    #           d’origine à assumer leurs responsabilités, soulignant des risques d’évasion à la faveur d’une éventuelle
    #            attaque de la Turquie sur le territoire, rendue possible par le retrait américain. Il a aussi précisé
    #             que les forces locales détenaient 800 hommes étrangers et retenaient 700 femmes et 1 500 enfants dans
    #              des camps de déplacés.\nLundi 18 février, les gouvernements européens qui avaient déjà engagé des
    #               discussions avec Washington au sujet du sort de leurs ressortissants ont été contraints de réagir
    #                dans l’urgence aux déclarations de Donald Trump."
    # print("One term SPACY", timeit.timeit(wrapper(match_syntagm_text_spacy, text, matcher), number=100))
    # print("One term custom", timeit.timeit(wrapper(match_syntagm_text_blob, "Donald Trump".split(), text), number=100))
    # matcher = PhraseMatcher(nlp.vocab)
    # matcher.add(1, None, nlp("Donald Trump"))
    # matcher.add(2, None, nlp("européens"))
    # matcher.add(3, None, nlp("combattre"))
    # matcher.add(4, None, nlp("Syrie"))
    # print("Spacy multiple term", timeit.timeit(wrapper(match_syntagm_text_spacy, text, matcher), number=100))
    # seqs = ["Donald Trump", "européens", "combattre", "syrie"]
    # seqs = [s.split() for s in seqs]
    # print(match_sequence("biopolymers".split(), "J'aime le paté".split()))
    # print("CUSTOM multiple term", timeit.timeit(wrapper(match_syntagm_text_blob_multi, seqs, text), number=100))

