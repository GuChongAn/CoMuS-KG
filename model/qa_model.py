import utils

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class qa_model():
    def __init__(self, template="qa_template"):
        self.template = utils.get_template(template)
        self.prompt = PromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            base_url="",
            api_key=""
        )
        self.parser = StrOutputParser()
        self.qa_chain = (
            self.prompt |
            self.llm |
            self.parser
        )
    
    def invoke_complete(self, triples):
        triples_complete = []
        for triple in triples:
            triples_complete.append(self.qa_chain.invoke(triple))
        return ",".join(triples_complete)

    def invoke_context(self, context, question):
        return self.qa_chain.invoke({"context": context, "question": question})