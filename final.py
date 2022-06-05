import codecs
from pykeen.datasets import Wikidata5M
import pandas as pd
import torch
from pykeen.datasets import get_dataset
import random

# get training dataset and load model
training = get_dataset(dataset=Wikidata5M).training
model = torch.load('..\\train\\Wikidata\\trained_model.pkl', map_location=torch.device('cpu'))


# Get relation from the Wikidata by str(relation)
def get_relation(relation):
    relation_id = ''
    df = codecs.open("wikidata5m_relation.txt", "r", "utf_8_sig")
    while True:
        line = df.readline()
        if not line:
            break
        a = (line.strip()).split("	")
        if relation in a:
            relation_id = a[0]
            break
    return relation_id


# Get entity from the Wikidata by str(entity)
def get_entity(entity):
    entity_id = ''
    df = codecs.open("wikidata5m_entity.txt", "r", "utf_8_sig")
    while True:
        line = df.readline()
        if not line:
            break
        a = (line.strip()).split("	")
        if entity in a:
            entity_id = a[0]
            break
    return entity_id


def get_random_triplet():
    file = codecs.open("wikidata5m_transductive_test.txt", "r", "utf_8_sig")
    randomnumber = random.randrange(1, 5133)
    i = 0
    triplet = 0
    while i != randomnumber:
        line = file.readline()
        a = (line.strip()).split("	")
        if i == randomnumber - 1:
            triplet = a
            break
        i += 1
    return triplet


def is_in_dataset(triplet):
    head = triplet[0]
    relation = triplet[1]
    tail = triplet[2]
    df = codecs.open("wikidata5m_all_triplet.txt", "r", "utf_8_sig")
    while True:
        line = df.readline()
        if not line:
            break
        a = (line.strip()).split("	")
        if head == a[0] and relation == a[1] and tail == a[2]:
            return True
    return False


# If hit in top-5 predictions, returns triple, else None
def quantity_in_dataset(df, triplet):
    num_of_rows = 5
    cols = df.columns
    iterable_entity = 1
    count = 0
    if cols[1] == 'tail_label':
        iterable_entity = 2
    elif cols[1] == 'head_label':
        iterable_entity = 0
    for i in range(num_of_rows):
        if iterable_entity == 0 or 2:
            triplet[iterable_entity] = df.iloc[i][cols[1]]
        else:
            triplet[iterable_entity] = df.iloc[i][cols[1]]
        count += is_in_dataset(triplet)
    return count


def get_relation_label(relation):
    relation_label = ''
    df = codecs.open("wikidata5m_relation.txt", "r", "utf_8_sig")
    while True:
        line = df.readline()
        if not line:
            break
        a = (line.strip()).split("	")
        if relation in a:
            relation_label = a[random.randrange(1, len(a))]
            break
    return relation_label


def get_entity_label(entity):
    df = codecs.open("wikidata5m_entity.txt", "r", "utf_8_sig")
    while True:
        line = df.readline()
        if not line:
            break
        a = (line.strip()).split("	")
        if entity in a:
            entity_label = a[random.randrange(1, len(a))]
            return entity_label


def get_first_hit(df, triplet):
    num_of_rows = 5
    cols = df.columns
    iterable_entity = 1
    if cols[1] == 'tail_label':
        iterable_entity = 2
    elif cols[1] == 'head_label':
        iterable_entity = 0

    for i in range(num_of_rows):
        triplet[iterable_entity] = df.iloc[i][cols[1]]
        if is_in_dataset(triplet):
            return triplet
    return None


def predict_relation(entities):
    new_triples = []
    for entity1 in entities:
        for entity2 in entities:
            if entity1 == entity2:
                continue

            head = entity1
            tail = entity2

            new_head = get_entity(head)
            new_tail = get_entity(tail)

            if new_tail != '' and new_head != '':
                predicted_relations_df = pd.DataFrame(
                    data=model.get_relation_prediction_df(head_label=new_head, tail_label=new_tail,
                                                          triples_factory=training),
                    columns=['relation_id', 'relation_label', 'score', 'in_training']
                )

                new_relation = get_first_hit(df=predicted_relations_df,
                                             triplet=[new_head, None, new_tail])
                if new_relation is not None:
                    # Find descriptions of predictions by wikidata library
                    new_triples.append([head, get_relation_label(new_relation[1]), tail])
                else:
                    continue

            # the same but swapped tail and head
            tail = entity1
            head = entity1

            new_head = get_entity(head)
            new_tail = get_entity(tail)

            if new_tail != '' and new_head != '':
                predicted_relations_df = pd.DataFrame(
                    data=model.get_relation_prediction_df(head_label=new_head, tail_label=new_tail,
                                                          triples_factory=training),
                    columns=['relation_id', 'relation_label', 'score', 'in_training']
                )

                # Find descriptions of predictions by wikidata library
                new_relation = get_first_hit(df=predicted_relations_df,
                                             triplet=[new_head, None, new_tail])

                if new_relation is not None:
                    # Find descriptions of predictions by wikidata library
                    new_triples.append([head, get_relation_label(new_relation[1]), tail])
                else:
                    continue
    return new_triples


def predict_head(entities, relations):
    new_triples = []
    for entity in entities:
        for relation in relations:

            relation = relation
            tail = entity

            new_relation = get_relation(relation)
            new_tail = get_entity(tail)

            if new_tail != '' and new_relation != '':
                predicted_heads_df = pd.DataFrame(
                    data=model.get_head_prediction_df(relation_label=new_relation, tail_label=new_tail,
                                                      triples_factory=training),
                    columns=['head_id', 'head_label', 'score', 'in_training']
                )

                new_head = get_first_hit(df=predicted_heads_df,
                                         triplet=[None, new_relation, new_tail])

                if new_head is not None:
                    new_triples.append([get_entity_label(new_head[0]), relation, tail])
                else:
                    continue

    return new_triples


def predict_tail(entities, relations):
    new_triples = []
    for entity in entities:
        for relation in relations:

            relation = relation
            head = entity

            new_relation = get_relation(relation)
            new_head = get_entity(head)
            if new_head != '' and new_relation != '':
                predicted_tails_df = pd.DataFrame(
                    data=model.get_tail_prediction_df(relation_label=new_relation, head_label=new_head,
                                                      triples_factory=training),
                    columns=['tail_id', 'tail_label', 'score', 'in_training']
                )

                new_tail = get_first_hit(df=predicted_tails_df,
                                         triplet=[new_head, new_relation, None])

                if new_tail is not None:
                    new_triples.append([head, relation, get_entity_label(new_tail[2])])
                else:
                    continue

    return new_triples


def evaluation():
    # For relation
    all_count = 0
    for i in range(10):
        random_triplet = get_random_triplet()
        predicted_relations_df = pd.DataFrame(
            data=model.get_relation_prediction_df(head_label=random_triplet[0], tail_label=random_triplet[2],
                                                  triples_factory=training),
            columns=['relation_id', 'relation_label', 'score', 'in_training'])
        count = quantity_in_dataset(predicted_relations_df, random_triplet)
        print(count, 'hits in ', i + 1, 'test prediction of relation')
        all_count += count
    print(all_count, 'hits in all tests prediction of relations')

    # For tail
    all_count = 0
    for i in range(10):
        random_triplet = get_random_triplet()
        predicted_tails_df = pd.DataFrame(
            data=model.get_tail_prediction_df(head_label=random_triplet[0], relation_label=random_triplet[1],
                                              triples_factory=training),
            columns=['tail_id', 'tail_label', 'score', 'in_training'])
        count = quantity_in_dataset(predicted_tails_df, random_triplet)
        print(count, 'hits in ', i + 1, 'test prediction of tail')
        all_count += count
    print(all_count, 'hits in all tests prediction of tails')

    # For head
    all_count = 0
    for i in range(10):
        random_triplet = get_random_triplet()
        predicted_heads_df = pd.DataFrame(
            data=model.get_head_prediction_df(relation_label=random_triplet[1], tail_label=random_triplet[2],
                                              triples_factory=training),
            columns=['head_id', 'head_label', 'score', 'in_training']
        )
        count = quantity_in_dataset(predicted_heads_df, random_triplet)
        print(count, 'hits in ', i + 1, 'test prediction of head')
        all_count += count
    print(all_count, 'hits in all tests prediction of heads')
