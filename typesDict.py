from typing import TypedDict

# inhertis from TypedDict
class Person(TypedDict):
    name : str
    age : int

# dictionary => define
per1: Person = {
    'name' : "Abhishek",
    'age' : 23
}