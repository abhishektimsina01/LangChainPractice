from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")  
model = "groq/compound-mini"

# document Loader
file_path = "Attention.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()
print(len(docs), type(docs))
# for i in docs:
#     print('-------------------------------------------------------')
#     print(i.page_content)
print('-------------------------------------------------------')
# print(docs[0])
print("-----------------------------------------------------------")
# spliting the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 80)
all_split = text_splitter.split_documents(docs)
print(all_split[0])

# defining the embedding model
embedding1 = OllamaEmbeddings(model="llama3.2")
embedding2 = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key="nk-C6hVY3u4YrKLu_eFYaVb3-TqP8ruu7gULm46gE8Utl0")
vector1 = embedding1.embed_query("I forgot my password")
vector2 = embedding2.embed_query("I forgot my password")
print("Dimension using ollama : ", len(vector1))
print("Dimension using nomic : ", len(vector2))
doc1 = ["I am a huge fan of Criastiano Ronaldo", "I love watching football", "I want to be an expert AI-Engineer"]
embedded_docs1 = embedding1.embed_documents(doc1)
embedded_docs2 = embedding2.embed_documents(doc1)
for i in embedded_docs1:
    print(i[:10])
vector3a = embedding1.embed_query("I play football")
vector3b = embedding2.embed_query("I play football")
ollama_score = cosine_similarity([vector3a], embedded_docs1)
nomic_score = cosine_similarity([vector3b], embedded_docs2)
print(ollama_score.max(), nomic_score.max())

# now we have to create vector store to store the vector embedding
vectoStore = Chroma.from_documents(
    documents=all_split,
    embedding=embedding1,
    collection_name="my_collection"
)

retriever = vectoStore.as_retriever(search_kwargs = {'k' : 3})

template = PromptTemplate(
    template="""The prompt is {query} and \n 
    {context}""",
    input_variables=['query', 'context']
)

query = input("Give me your question")

relevant_chunks = retriever.invoke(query)
page_content = []
for i in relevant_chunks:
    page_content.append(i.page_content)
prompt = template.invoke({'query' : query, 'context' : page_content})
print(prompt)

llm = ChatGroq(
    model = "groq/compound-mini",
    api_key = api_key
)

response = llm.invoke(prompt)
print(response.content)