# tools are a block of logic that can execute a cerain task.
# tools are used by agents in order to perform action decided by the LLM
# AI-Agents = LLM(reasoning and decision-making capability) and Tools(action)
# tools are always runnable (we can invoke the tool by passing dictionary)

# for custom tools
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import requests
from typing import Annotated
load_dotenv()

print("Conversion tool")

llm = ChatGroq(
    model ="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key")
)

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