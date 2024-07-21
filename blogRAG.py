import os
import env
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set env variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Load blog
loader = WebBaseLoader(
    web_paths=("https://vignesh.page/posts/kafka/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("main-content")
        )
    ),
)
blog_docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("What is Kafka?")

# LLM
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Chain
chain = prompt | llm

# Run
chain.invoke({"context":docs,"question":"What is Kafka?"})



### Some Syntatic Sugar
prompt_hub_rag = hub.pull("rlm/rag-prompt")


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_hub_rag
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Kafka?")