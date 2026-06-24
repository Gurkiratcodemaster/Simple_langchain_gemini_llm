from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0.7
)
notes_prompt = ChatPromptTemplate.from_template(
    "Create Study Notes about {topic} in bullet points."
)
quiz_prompt = ChatPromptTemplate.from_template(
    "Create a Quiz from {notes} and in heading write {topic} Quiz"
)
chain = (
    RunnablePassthrough.assign(
        notes = notes_prompt | llm | StrOutputParser()
    ) |
    quiz_prompt | llm | StrOutputParser()
)
response = chain.invoke({"topic": "Agentic AI"})
print(response)
