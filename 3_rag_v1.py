# pip install -U langchain langchain-groq langchain-community faiss-cpu pypdf python-dotenv
# pip install langchain langchain-ollama langchain-community chromadb

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = 'RAG CHATBOT'
PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page


# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False)
splits = splitter.split_documents(docs)


# 3) Embed + index
emb = OllamaEmbeddings(model="nomic-embed-text")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful PDF question-answering assistant.

Use ONLY the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided PDF."

Be concise and accurate."""
    ),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# 5) Chain
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.5)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


chain = parallel | prompt | llm | StrOutputParser()


# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
