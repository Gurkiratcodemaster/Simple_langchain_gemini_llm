from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0.7
)
notes_prompt = ChatPromptTemplate.from_template(
    "Create Study Notes about {topic} in bullet points."
)
notes_chain = notes_prompt | llm | StrOutputParser()
notes = notes_chain.invoke({"topic":"Agentic AI"})
quiz_prompt = ChatPromptTemplate.from_template(
    "Create a Quiz from {notes}"
)
quiz_chain = quiz_prompt | llm | StrOutputParser()
response = quiz_chain.invoke({"notes": notes})
print(response)
