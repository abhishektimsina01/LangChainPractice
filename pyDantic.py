from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from typing import Literal, Optional
from dotenv import load_dotenv
load_dotenv()
try:
    class Person(BaseModel):
        name : str = Field(description= "This contains the name of the person")
        age : int = Field(gt=18, lt=20, description="this contains the age of the person")

    parser = PydanticOutputParser(pydantic_object=Person)
    parser1 = StrOutputParser()
    template = PromptTemplate(
        template="Give me the name and age for an random person who is {personal_data} {format_instructions}",
        input_variables=["peronal_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = ChatGroq(
        model="groq/compound-mini",
        api_key=os.getenv("api_key")
    )
    prompt = template.invoke({'personal_data' : 'indian'})
    # print(prompt)
    response = template | llm | parser
    print(response.invoke({'personal_data' : 'indian'}))
except Exception as e:
    print(e)