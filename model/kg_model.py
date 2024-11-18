import utils

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

class kg_model():
    def __init__(self):
        # large language model and Output Parser
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            base_url="",
            api_key=""
        )
        self.parser = StrOutputParser()

        # question decomposition langchain 
        self.qd_template = utils.get_template("question_decomposition_template")
        self.qd_template = PromptTemplate.from_template(self.qd_template)
        self.qd_chain = (
            self.qd_template |
            self.llm |
            self.parser
        )

        # simple questions -> uncomplete KG triple Langchain
        self.q2t_template = utils.get_template("questions2triple_template")
        self.q2t_template = PromptTemplate.from_template(self.q2t_template)
        self.q2t_chain = (
            self.q2t_template |
            self.llm |
            self.parser
        )

        # embedding model assit KG retriever
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def invoke(self, question, graph, embeddings, isKG=True):
        # question decomposition and converts to KG triples
        questions = self.qd_chain.invoke({"question": question}).split('\n')
        uncomplete_triples = []
        for q in questions:
            t = self.q2t_chain.invoke({"question": q})
            uncomplete_triples.append(t)
        if not isKG:
            return uncomplete_triples

        # Uncomplete triples completion by Knowledge Graph
        graph = utils.build_graph(graph, embeddings)
        triples = []
        for triple in uncomplete_triples:
            # preprocessing
            triple = triple.strip('()').split(',')
            type = utils.check_uncomplete_type(triple)
            if type == 0:
                triple = utils.uncomplete_head2tail(triple)
                type = 2
            # triple completion
            triple = utils.complete_triple(triple, type, graph, self.embedding_model)
            if isinstance(triple[0], list):
                triples += triple
            else:
                triples.append(triple)
        
        return triples