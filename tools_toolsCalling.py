# tools are a block of logic that can execute a cerain task.
# tools are used by agents in order to perform action decided by the LLM
# AI-Agents = LLM(reasoning and decision-making capability) and Tools(action)
# tools are always runnable (we can invoke the tool by passing dictionary)

# tools = built-in tools and custom tools
from langchain_community.tools import DuckDuckGoSearchRun
# for custom tools
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# built-in tools
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("Abhishek Timsina")
# print(results)

print("Custom tools")
@tool
def multiplyTool(a : int, b : int) -> int:
    # docs string => llm can understand waht the function can really do
    """given 2 nunbers a and b, it returns the multiply of those those two number"""
    return a*b

result = multiplyTool.invoke({'a': 5, 'b':5})
print(result)

print(multiplyTool.name)
print(multiplyTool.description)
print(multiplyTool.args)

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
print(multiply_tool.invoke({'a' : 1, 'b' : 1}))


# tool binding = connecting llm and tool
llm = ChatGroq(
    model ="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key")
)

llm_with_tools = llm.bind_tools([multiplyTool])
print(llm_with_tools.invoke("Hi, how are you?"))
print("---------------------------------------------")

# it just suggest us the tool we can use and also the arguements we have to pass while using tools
query = HumanMessage("can you multiply 3 with 10")
messages = [query]
result = llm_with_tools.invoke(messages)
messages.append(result)
tool_result = multiplyTool.invoke(result.tool_calls[0])
messages.append(tool_result)
print(messages)

print(llm_with_tools.invoke(messages).content)