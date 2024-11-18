import utils
import os
import uuid
from datasets import load_dataset

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class multimodal_rag_model():
    def __init__(self):
        # large language model and Output Parser
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            base_url="",
            api_key=""
        )
        self.parser = StrOutputParser()

        # text summary langchain 
        self.text_summary_template = utils.get_template("text_summary_template")
        self.text_summary_prompt = PromptTemplate.from_template(self.text_summary_template)
        self.text_summary_chain = (
            self.text_summary_prompt |
            self.llm |
            self.parser
        )

        # multimodal RAG langchain
        self.multimodal_rag_template = utils.get_template("multimodal_rag_template")
        self.multimodal_rag_prompt = PromptTemplate.from_template(self.multimodal_rag_template)
        self.multimodal_rag_chain = (
            self.multimodal_rag_prompt |
            self.llm |
            self.parser
        )

        # rag retriever
        self.vectorstore = Chroma(
            collection_name="test", 
            embedding_function=HuggingFaceBgeEmbeddings(), 
            persist_directory="C:/Users/86181/Desktop/MKGC/src/data/chroma.db"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        ## self.init_retriever("C:/Users/86181/Desktop/MKGC/dataset/image") # Only should init first

    def init_retriever(self, image_path):
        # load text data
        wikipedie_dataset = load_dataset("philschmid/easyrag-mini-wikipedia", "documents", split="full")

        # text summary
        texts = []
        text_summaries = []
        i = 0
        for doc in wikipedie_dataset:
            texts.append(doc['document'])
            text_summary = self.text_summary_chain.invoke(doc['document'])
            text_summaries.append(text_summary)
            i += 1
            if i == 10:
                break
            
        # image summary
        image_summaries = []
        image_list = os.listdir(image_path)
        for image_name in image_list:
            image_summary = utils.image_summary(os.path.join(image_path, image_name))
            image_summaries.append(image_summary)

        # init multi vector retriever
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={"doc_id": doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)

        add_documents(self.retriever, text_summaries, texts)
        add_documents(self.retriever, image_summaries, image_summaries)

    def tmp(self, text):
        self.retriever.invoke(text)

    def invoke(self, triples):
        # triples preprocess
        triple_text = ""
        for triple in triples:
            tmp = ",".join(triple)
            tmp = "(" + tmp + ")" + ","
            triple_text += tmp
        
        # get retriever knowledge
        context = ""
        for doc in self.retriever.invoke(triple_text):
            context += doc.page_content 
        
        # returen rag result
        return self.multimodal_rag_chain.invoke({"context": context, "triple": triple_text})