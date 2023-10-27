from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

loader =  OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
data = loader.load()
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[30].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

#pinecone
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'asia-southeast1-gcp') 

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV 
)
index_name = "" 

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

docsearch = Chroma.from_documents(texts, embeddings)
query = "What are examples of good data science teams?"
docs = docsearch.similarity_search(query)

print(docs[0].page_content[:450])

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
query = "What is the collect stage of data maturity?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)