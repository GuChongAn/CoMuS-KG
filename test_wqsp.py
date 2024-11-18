from datasets import load_dataset
from tqdm import tqdm
import warnings

from model.kg_model import kg_model
from model.qa_model import qa_model
from model.multimodal_rag_model import multimodal_rag_model

import utils
warnings.filterwarnings('ignore')

# load dataset WebQSP
# dataset_wqsp = load_dataset("rmanluo/RoG-webqsp", split='test')
dataset_wqsp = load_dataset("json", data_files="C:/Users/86181/Desktop/MKGC/src/data/wqsp_sub_tmp.json", split='train')

# init model
kg_model = kg_model()
multimodal_rag_model = multimodal_rag_model()
qa_model = qa_model()

i = 0
# test loop
for data in tqdm(dataset_wqsp):
    # Get triple in Knowledge Graph
    triples = kg_model.invoke(data['question'], data['graph'], data['embeddings'])
    
    # Complete triple by multimodal RAG
    triples_rag = multimodal_rag_model.invoke(triples)
    
    # Answer question based triple
    answer = qa_model.invoke(triples_rag, data['question'])
        
    with open("C:/Users/86181/Desktop/MKGC/src/res/wqsp_gpt3.5_my.txt", 'a+') as f:
        f.write("{}\t{}\t{}\n".format(i, data['question'], answer))

    i += 1
    break