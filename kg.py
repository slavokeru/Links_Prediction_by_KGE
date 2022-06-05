from openie import StanfordOpenIE
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def make_lists_of_text(data):
    """part of preprocessing"""
    with open(data) as f:
        list_of_text = []
        text = f.read()
        list_of_text.append(text.split('.'))
    return list_of_text


def kg(list_of_sentences, subjects=None, relations=None, objects=None):
    if objects is None:
        objects = []
    if relations is None:
        relations = []
    if subjects is None:
        subjects = []
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    triple_list = []
    with StanfordOpenIE(properties=properties) as client:
        for sentence in list_of_sentences[0]:
            for triple in client.annotate(sentence):
                head = triple['subject']
                relation = triple['relation']
                tail = triple['object']
                triple_list.append([head, relation, tail])

                subjects.append(triple['subject'])
                relations.append(triple['relation'])
                objects.append(triple['object'])

    return subjects, relations, objects, triple_list


def draw_graph(subjects, relations, objects):
    kg_df = pd.DataFrame({'source': subjects, 'target': objects, 'edge': relations})

    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph(), edge_key="edge")

    plt.figure(figsize=(12, 12))

    triples = {}
    for i in range(len(subjects)):
        triples[(subjects[i], objects[i])] = relations[i]
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, triples)

    plt.show()
