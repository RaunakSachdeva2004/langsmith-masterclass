from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'



prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
