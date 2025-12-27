from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")
model = "groq/compound-mini"

# we need to load the pdf from the file