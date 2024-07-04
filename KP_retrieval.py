import numpy as np
import pandas as pd
import spacy
import os
import json

import urllib
from argparse import ArgumentParser, Namespace
import argparse


from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
import networkx as nx
import matplotlib.pyplot as plt

import requests
import hashlib
import os
import atexit
from functools import lru_cache
import time
import logging
from collections import defaultdict, deque
# import queue

from pywikidata.pywikidata import Entity
from collections import Counter

from config import *

global_nlp = spacy.load("en_core_web_sm")
sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@lru_cache(maxsize=None)
def wiki_search_entities(query = 'Trump'):
    try:
        service_url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbsearchentities',
            'search': query,
            'language': 'en',
            'limit': 20,
            'format': 'json'
        }
        print("searching", query, '...')
        url = service_url + '?' + urllib.parse.urlencode(params)
        response = json.loads(urllib.request.urlopen(url).read())
    except Exception as e:
        print(str(e))
        return {}
    return response


def get_limit_text(text, limit = 4096):
    cnt = 0
    for i in range(len(text)):
        if text[i] in ['\t', '\n', ' ']:
            cnt += 1
        if cnt >= limit:
            return text[:i]
    return text

def get_entities_from_text(text, return_type='dict'):
    res = {}
    if text.strip() == '':
        return json.dumps(res)
    doc = global_nlp( get_limit_text(text) )
    for entity in doc.ents:
        if entity.label_ in ['DATE', 'TIME', 'MONEY', 'CARDINAL', 'TIME', 'PERCENT', 'ORDINAL', 'QUANTITY', 'LANGUAGE']:
            continue
        if entity.text not in res:
            res[entity.text] = entity.label_
    if return_type == 'str':
        return json.dumps(res)
    elif return_type == 'dict':
        return res
    else:
        return res


def knowledge_path_prob_score(r, e, query):
    # s_p = ', '.join([e.label for e in p])
    vec_p = sent_model.encode(query, convert_to_tensor=True)
    vec_e = sent_model.encode(r.label + ', ' + e.label, convert_to_tensor=True)

    s_val = np.dot(vec_p.detach().cpu().numpy(), vec_e.detach().cpu().numpy().T)
    beta = 0.6
    probability = 1 / (1 + np.exp(beta - s_val))
    print("probability {}, {}: {}".format(r, e, probability))
    return probability

def knowledge_path_retrieval(qid, query):
    path_list = []

    entity = Entity(qid)  # e.g. Q90
    V = []
    Q = deque( [(entity, [entity])] )
    while Q:
        (cur, path) = Q.popleft()
        V.append(cur)

        # forward expansion
        print('---- forward_one_hop_neighbours')
        for property, next in cur.forward_one_hop_neighbours:  # >> [(<Entity(Property): P6>, <Entity: Q2851133>), (<Entity(Property): P8138>, <Entity: Q108921672>), ...]
            print(property.idx, property.label, ' --> ', next.idx, next.label)
            if property.label is None or next.label is None:
                continue
            if next.idx not in V and len(path) < 5: # 确保右边不超过 2 个
                score  = knowledge_path_prob_score(property, next, query)
                if score > 0.5:
                    path = path + [property, next]
                    path_list.append((path, score))
                    Q.append((next, path))
                    # V.append(next)

        # backward expansion
        print('---- backward_one_hop_neighbours')
        for property, next in entity.backward_one_hop_neighbours:
            print(property.idx, property.label, ' <-- ', next.idx, next.label)
            if property.label is None or next.label is None:
                continue
            if next.idx not in V and len(path) < 9: # 确保左右两边各不超过 2 个。总和 5 个 entities， 4 个 relations
                score = knowledge_path_prob_score(property, next, query)
                if score > 0.5:
                    path = [next, property] + path
                    path_list.append((path, score))
                    Q.append((next, path))
                    # V.append(next)

    path_list.sort(key=lambda x: x[1], reverse=True)

    res = []
    for p, s in path_list:
        res.append({
            'path': [e.label for e in p],
            'score': s
        })
        print(s, '-->'.join([e.label for e in p]))
    return res

