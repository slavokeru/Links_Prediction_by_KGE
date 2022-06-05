import kg
import final as kge

list_of_text = kg.make_lists_of_text(data='Data.txt')

relations = []
subjects = []
objects = []

subjects, relations, objects, triples = kg.kg(list_of_text, subjects, relations, objects)

print(f'got triples: {triples}')
kg.draw_graph(subjects, relations, objects)

entities = set(subjects + objects)
relations_set = set(relations)

print(f'entities and relations: {entities, relations}')

new_triples = kge.predict_head(entities=entities, relations=relations_set)

print(f'got new triples: {new_triples}')

if len(new_triples):
    for triple in new_triples:
        relations.append(triple[1])
        subjects.append(triple[0])
        objects.append(triple[2])


new_triples = kge.predict_relation(entities=entities)

print(f'got new triples: {new_triples}')

if len(new_triples):
    for triple in new_triples:
        relations.append(triple[1])
        subjects.append(triple[0])
        objects.append(triple[2])

new_triples = kge.predict_tail(entities=entities, relations=relations_set)

print(f'got new triples: {new_triples}')

if len(new_triples):
    for triple in new_triples:
        relations.append(triple[1])
        subjects.append(triple[0])
        objects.append(triple[2])

kg.draw_graph(subjects, relations, objects)

print('\n\nDo you want to run evaluation? [y/n]', end='')
evaluation = input()

if evaluation == 'y':
    print(kge.evaluation())
else:
    print('evaluation skipped')
