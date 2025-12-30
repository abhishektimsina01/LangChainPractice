from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(
    model="groq/compound-mini",
    api_key=os.getenv("api_key")
)

parser = StrOutputParser()

template = PromptTemplate(template="Give me a short summary on {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Give me a short joke on {topic}", input_variables=["topic"])
template1 = PromptTemplate(template="Give me a short explainattion of the joke {fun}", input_variables=["fun"])

prev_chain = template2 | llm | parser

parallel_chain = RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "explaination" : RunnableSequence(template1, llm, parser)
    }
)

after_chain = prev_chain | parallel_chain
print(after_chain.invoke({'topic' : "Abhi"}))

# runnable = RunnableSequence(template, llm, parser)
# print(runnable.invoke({'topic' : 'Natural Language'}))
