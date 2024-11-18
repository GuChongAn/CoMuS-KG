import json
import requests
from tqdm import tqdm

import networkx as nx

def get_template(template_name):
    with open("C:/Users/86181/Desktop/MKGC/src/config/template.json", 'r') as f:
        template_json = json.load(f)
    return template_json[template_name]

def image_summary(image_path):
    file = {'file': open(image_path, 'rb')}
    response = requests.post(
        url='http://124.220.33.102:8085/image_chat', 
        files=file, 
        data={'query': 'please briefly summary this image'})
    return response.text

# Utils about knowledge graph process
# build graph by triple list
def build_graph(graph, embeddings):
    G = nx.Graph()
    for i, triplet in tqdm(enumerate(graph)):
        h, r, t = triplet
        h_embedding, r_embedding, t_embedding = embeddings[i]
        G.add_node(h, embeddings=h_embedding)
        G.add_node(t, embeddings=t_embedding)
        G.add_edge(h, t, relation=r.strip(), embeddings=r_embedding)
    return G

# Check the triplet missing relation or entity
def check_uncomplete_type(triple):
    head = triple[0]
    relation = triple[1]
    if '?' in head:
        return 0
    elif '?' in relation:
        return 1
    else:
        return 2
    
# Convert missing head to missing tail
def uncomplete_head2tail(triple):
    return [triple[2], triple[1], triple[0]]

# Get similarity in input entity and candidate enetity list
def get_similarity_entity(input_embeddings, enetity_list, graph):
    candidate = []
    for entity in enetity_list:
        tmp_embeddings = graph.nodes[entity]['embeddings']
        tmp_similarity = tmp_embeddings @ input_embeddings
        if tmp_similarity > 0.9:
            candidate.append(entity)
    return candidate

# Complete triples
def complete_triple(triple, type, graph, embedding_model):
    # return triples
    tmp_triples = []
        
    # get triple embeddings
    h_embeddings = embedding_model.encode(triple[0])
    r_embeddings = embedding_model.encode(triple[1])
    t_embeddings = embedding_model.encode(triple[2])
    
    # tail missing
    if type == 2:
        # get head candidate
        node_list = list(graph.nodes)
        head_list = get_similarity_entity(h_embeddings, node_list, graph)
        
        # check head candidate's relation
        for head in head_list:
            adj_list = list(graph[head])
            for adj in adj_list:
                tmp_embeddings =  graph.edges[head, adj]['embeddings']
                tmp_similarity = tmp_embeddings @ r_embeddings
                if tmp_similarity > 0.8:
                    tmp_triples.append([head, graph.edges[head, adj]['relation'], adj])
    
    # relation missing
    elif type == 1:
        # get head candidate
        node_list = list(graph.nodes)
        head_list = get_similarity_entity(h_embeddings, node_list, graph)

        # get tailed candidate
        tail_list = get_similarity_entity(t_embeddings, node_list, graph)

        # check relation's embeddings
        for head in head_list:
            adj_list = list(graph(head))
            for adj in adj_list:
                if adj in tail_list:
                    tmp_triples.append([head, graph.edges[head, adj]['relation'], adj])
    
    if tmp_triples:
        return tmp_triples

    return triple

def display(question, ref_answer, triples, complete_triples, output):
    return print("\nquestion: {}\nanswer: {}\nuncomplete triples: {}\ncomplete triples: {}\nmodel output: {}".format(question, ref_answer, triples, complete_triples, output))