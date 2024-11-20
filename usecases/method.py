from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import rdflib
from rdflib import Graph, URIRef, Literal
import re
import editdistance
import pickle
import os
import time
import llama
import csv
import subprocess
import numpy as np
import sentence_BERT


# def find_label(s):
#     label_space = 'http://www.w3.org/2000/01/rdf-schema#label'
#     query_template_label = "SELECT DISTINCT ?x WHERE {{<{}> <{}> ?x.}}".format(s, label_space)
    
#     qres = g.query(query_template_label)
#     ass = 'Label not found'
#     for row in qres:
#         ass = str(row.x)
#         break
#     return ass

g = Graph()
g.parse('/home/tongsu/Speakeasy-agent/data/14_graph.nt', format='turtle')

node_labels = {}
predicate_labels = {}
path1 = '/home/tongsu/Speakeasy-agent/data/node_labels.pkl'
path2 = '/home/tongsu/Speakeasy-agent/data/predicate_labels.pkl'
with open(path1, 'rb') as f:
    node_labels = pickle.load(f)
with open(path2, 'rb') as f:
    predicate_labels = pickle.load(f)


def template_1(question):
    # # process input
    # sent = process_input(question)
    # predicted_labels = predict_entities(sent, crf)

    # print("Input Text:", question)

    # entity = extract_entities(sent, predicted_labels)

    # entity = ' '.join(entity)

    # #defining a question pattern
    # question_pattern = "who is the (.*) of ENTITY"

    # question = re.sub(entity, "ENTITY", question.rstrip("?"))  # preprocess the question

    # relation = re.match(question_pattern, question).group(1)  # match the relation using a pattern
    
    entity, relation = llama.entity_relation_extract(question)
    
    relation = sentence_BERT.get_closest(relation)
    
    print(relation)

    #匹配
    tmp = 9999
    match_node = ""
    for key, label in node_labels.items():
        if editdistance.eval(label, entity) < tmp:
            tmp = editdistance.eval(label, entity)
            match_node = key

    tmp = 9999
    match_pred = ""
    for key, label in predicate_labels.items():
        if editdistance.eval(label, relation) < tmp:
            tmp = editdistance.eval(label, relation)
            match_pred = key

    # print("\n--- the matching node of \"{}\" is {}\n".format(entity, match_node))
    # print("--- the matching predicates of \"{}\" is {}\n".format(relation, match_pred))

    #query
    query_template = "SELECT DISTINCT ?x ?y WHERE {{ <{}> <{}> ?x.  }}".format(match_node, match_pred )

    qres = g.query(query_template)
    # print("\n--- querying results: ")
    answer = 'oops no answer'
    for row in qres:
        # print(row.x)
        print(type(row.x.toPython()))
        try:
            answer = node_labels[row.x.toPython()]
        except KeyError:
            answer = row.x
    # print("answer is:",format(answer))



    WD = rdflib.Namespace('http://www.wikidata.org/entity/')
    WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
    DDIS = rdflib.Namespace('http://ddis.ch/atai/')
    RDFS = rdflib.namespace.RDFS
    SCHEMA = rdflib.Namespace('http://schema.org/')
    # load the embeddings
    
    entity_emb = np.load("/home/tongsu/Speakeasy-agent/data/embeddings/entity_embeds.npy")
    relation_emb = np.load("/home/tongsu/Speakeasy-agent/data/embeddings/relation_embeds.npy")
    entity_file = "/home/tongsu/Speakeasy-agent/data/embeddings/entity_ids.del"
    relation_file = "/home/tongsu/Speakeasy-agent/data/embeddings/relation_ids.del"
    # load the dictionaries
    with open(entity_file, 'r') as ifile:
        ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
        id2ent = {v: k for k, v in ent2id.items()}
    with open(relation_file, 'r') as ifile:
        rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
        id2rel = {v: k for k, v in rel2id.items()}
    ent2lbl = {ent: str(lbl) for ent, lbl in g.subject_objects(RDFS.label)}
    lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

    # Find the Wikidata ID for the movie (https://www.wikidata.org/wiki/Q132863 is the ID for "Finding Nemo")
    movie = WD[match_node.split('/')[-1]]

    # Find the movie in the graph
    movie_id = ent2id[movie]

    # we compare the embedding of the query entity to all other entity embeddings
    distances = pairwise_distances(entity_emb[movie_id].reshape(1, -1), entity_emb, metric='cosine').flatten()

    # and sort them by distance
    most_likely = distances.argsort()

    for rank, idx in enumerate(most_likely[:3]):
        rank = rank + 1
        ent = id2ent[idx]
        q_id = ent.split('/')[-1]
        lbl = ent2lbl[ent]
        dist = distances[idx]

    movie_emb = entity_emb[ent2id[movie]]

    try:
        # Find the predicate (relation) of the genre (https://www.wikidata.org/wiki/Property:P136 is the ID for "genre")
        genre = WDT[match_pred.split('/')[-1]]
        genre_emb = relation_emb[rel2id[genre]]

        # combine according to the TransE scoring function
        lhs = movie_emb + genre_emb

        # compute distance to *any* entity
        distances = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)

        # find most plausible tails
        most_likely = distances.argsort()

        embedding_list = []

        # show most likely entities
        for rank, idx in enumerate(most_likely[:3]):
            rank = rank + 1
            ent = id2ent[idx]
            q_id = ent.split('/')[-1]
            lbl = ent2lbl[ent]
            dist = distances[idx]
            embedding_list.append(lbl)
            
        answer_embed = ','.join(embedding_list)
    
    except:
        answer_embed = "oops no answer"

    #answer
    if (answer != "oops no answer") and (answer_embed != "oops no answer"):
        answer_template = "Hi, the factual answer is: {}, ".format(answer) + "and the answers suggested by embeddings are: {}.".format(answer_embed)
        
    elif answer == "oops no answer":
        answer_template = "Hi, the answers suggested by embeddings are: {}.".format(answer_embed)
    
    elif answer_embed == "oops no answer":
        answer_template = "Hi, the factual answer is: {}.".format(answer)
    
    else:
        answer_template = "Apologies, but there is no corresponding answer in the database for your question."
        

    # print("\n--- generated response: {}".format(answer_template))
    return answer_template

if __name__ == '__main__':
    Input = "Who is the director of Star Wars: Episode VI - Return of the Jedi?"
    print(template_1(Input))