from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def get_embeddings(graph, embedding_model):
    graph_embeddings = []
    for triplet in tqdm(graph):
        h, r, t = triplet
        h_embedding = embedding_model.encode(h)
        r_embedding = embedding_model.encode(r)
        t_embedding = embedding_model.encode(t)
        graph_embeddings.append([h_embedding, r_embedding, t_embedding])
    return graph_embeddings

def add_embeddings_column(dataset, embedding_model):
    dataset['embeddings'] = get_embeddings(dataset['graph'], embedding_model)
    return dataset

# graph embedding and store
if __name__ == "__main__":
    # load dataset WebQSP
    dataset_wqsp = load_dataset("json", data_files="C:/Users/86181/Desktop/MKGC/src/data/wqsp_sub.json", split='train')

    # embeddinsg model
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # process dataset
    dataset_wqsp_sub = dataset_wqsp.map(add_embeddings_column, fn_kwargs={'embedding_model': embedding_model})
    
    # srote dataset
    dataset_wqsp_sub.to_json("C:/Users/86181/Desktop/MKGC/src/data/wqsp_sub_tmp.json")
