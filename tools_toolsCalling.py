# tools are a block of logic that can execute a cerain task.
# tools are used by agents in order to perform action decided by the LLM
# AI-Agents = LLM(reasoning and decision-making capability) and Tools(action)
# tools are always runnable (we can invoke the tool by passing dictionary)

# tools = built-in tools and custom tools
from langchain_community.tools import DuckDuckGoSearchRun
# for custom tools
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import requests
from typing import Annotated
load_dotenv()

# built-in tools
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("Abhishek Timsina")
# print(results)

# print("Custom tools")
@tool
def multiplyTool(a : int, b : int) -> int:
    # docs string => llm can understand waht the function can really do
    """given 2 nunbers a and b, it returns the multiply of those those two number"""
    return a*b

result = multiplyTool.invoke({'a': 5, 'b':5})
# print(result)

# print(multiplyTool.name)
# print(multiplyTool.description)
# print(multiplyTool.args)

class MultiplyInput(BaseModel):
    a : int = Field(description="first number to add", required = True)
    b : int = Field(description="second number to add", required = True)

def multiply(a : int, b : int) -> int:
    return a*b

# creating tool using structuredTool
multiply_tool = StructuredTool.from_function(
    func=multiply,
    name = "muiltiply",
    description="multiply two number",
    args_schema=MultiplyInput
)
# print(multiply_tool.invoke({'a' : 1, 'b' : 1}))


# tool binding = connecting llm and tool
llm = ChatGroq(
    model ="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key")
)

llm_with_tools = llm.bind_tools([multiplyTool])
# print(llm_with_tools.invoke("Hi, how are you?"))
print("---------------------------------------------")

# it just suggest us the tool we can use and also the arguements we have to pass while using tools
query = HumanMessage("can you multiply 3 with 10")
messages = [query]
result = llm_with_tools.invoke(messages)
print(result)
messages.append(result)
tool_result = multiplyTool.invoke(result.tool_calls[0])
print(tool_result)
messages.append(tool_result)
print(messages)
print(llm_with_tools.invoke(messages).content)  


print("------------------------------------------------------")
print("Conversion tool")

# tools for LLM
@tool
def Conversion(baseCurrency: str, targetCurrency : str) -> float:
    '''Given the baseCurrency and targetCurrency, we call the api for the currency conversion factor right now between them'''
    url = f"https://v6.exchangerate-api.com/v6/{os.getenv("currency_api_key")}/latest/{baseCurrency}"
    response = requests.get(url)
    return response.json()['conversion_rates'][targetCurrency]

@tool
def CurrencyExchange(baseCurrency : float, conversionRate : Annotated[float, InjectedToolArg]) -> float:
    '''coverts the given baseCurrency by multiplying two input values baseCurrency and conversionRate to calculate the target currency value'''
    return baseCurrency*conversionRate


conversionRate = Conversion.invoke({'baseCurrency': "USD", 'targetCurrency' : "NPR"})
print(conversionRate)
print(CurrencyExchange.invoke({'baseCurrency': 10, 'conversionRate' : conversionRate}))

llm_with_tools1 = llm.bind_tools([Conversion, CurrencyExchange])

message = [HumanMessage("What is the fullform OF LLM?")]
print(llm_with_tools1.invoke(message))

message = [HumanMessage("What is the currency conversion factor of USD to NPR and convert 10 USD into NPR")]
aiMessage = llm_with_tools1.invoke(message)
message.append(aiMessage)

for tool_call in aiMessage.tool_calls:
    print(tool_call)
    if tool_call['name'] == "Conversion":
        toolMessage = Conversion.invoke(tool_call)
        message.append(toolMessage)
    if tool_call['name'] == "CurrencyExchange":
        tool_call['args']['conversionRate'] = float(toolMessage.content)
        toolMessage = CurrencyExchange.invoke(tool_call)
        message.append(toolMessage)

print(message)
print(llm_with_tools1.invoke(message).content)