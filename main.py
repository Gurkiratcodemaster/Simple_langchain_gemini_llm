from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature = 0.7
)
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in Simple Terms"
)
chain = prompt | llm
response = chain.invoke({"topic":"Agentic AI"})
print(response.content[0]["text"])
