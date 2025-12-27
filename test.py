from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_nomic import NomicEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Optional
import os
load_dotenv()

class feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description= "This contains theh positive or negative value baed upon the sentiment of the feeback from the user")

parser1 = PydanticOutputParser(pydantic_object=feedback)    

# embeddings = OllamaEmbeddings(model="llama3.2")
# embeddings2 = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key="nk-C6hVY3u4YrKLu_eFYaVb3-TqP8ruu7gULm46gE8Utl0")
# vector5 = embeddings.embed_documents(["Hello my name is Abhishek Timsina", "I am currntly studying CSIT in Texas International college", "I love eating MO:MO"])
# vector6 = embeddings.embed_query("Myself")
# print(cosine_similarity([vector6], vector5))



template1 = PromptTemplate(template="Write a detailed report on the {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Give me a 5 line summary on the report {report} znd make it visible in the form of point lie 1,2,3", input_variables=["report"])
template3 = PromptTemplate(template="What are the {topic} in 5 lines", input_variables=["topic"])
template6 = PromptTemplate(template="What are the {topic1} in 5 lines", input_variables=["topic1"])
template4 = PromptTemplate(template="give me the mergesd summary of the two answers:  {ans1} and {ans2}", input_variables=["ans1", "ans2"])
template5 = PromptTemplate(template= "Give me the sentiment of the feedback as : {feedback} such that, {format}", 
                           input_variables=["feedback"],
                           partial_variables= {"format" : parser1.get_format_instructions()})

parser = StrOutputParser()

llm1 = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("api_key"),
    temperature=0
)

llm2 = ChatGroq(
    model="groq/compound-mini",
    api_key=os.getenv("api_key"),
    temperature=0
)

# chain for the parallel chain
try:
    parallel_chain = RunnableParallel({
        'ans1' : template3 | llm1 | parser, # output1
        'ans2' : template6 | llm2 | parser  # output2
    })

    # chain after the combination
    after_merge_chain = template4 | llm1 | parser

    # chain from the start to the end
    total_chain = parallel_chain | after_merge_chain

    response = total_chain.invoke({"topic1" : "Encoder", 'topic' : "Encoder"})
    # chain = template1 | llm1 | parser | template2 | llm1 | parser
    # response = chain.invoke({"topic" : "Encoder"})  
    # response = llm.invoke("HELLO!!")
    print(response)
    print(total_chain.get_graph().print_ascii())

    classification = template5 | llm1 | parser1
    print(classification.invoke({'feedback' : "this is the best item i have ever seen."}))

    prompt_positive = PromptTemplate(template="give me one response for the following positive feedback : {feedback}",
                                    input_variables=["feedback"])
    prompt_negative = PromptTemplate(template="give me one response for the following negative feedback : {feedback}",
                                    input_variables=["feedback"])
    print("Conditional chain start")
    branch_chain = RunnableBranch(
        (lambda x:x.sentiment == 'positive', prompt_positive | llm1 | parser),
        (lambda x:x.sentiment == 'negative', prompt_negative | llm1 | parser),
        RunnableLambda(lambda x: "couldn't find sentix.sentiment")
    )       
    chain = classification | branch_chain
    print(chain.invoke({'feedback' : "love it"}))
    chain.get_graph().print_ascii()
except Exception as e: 
    print(e)


