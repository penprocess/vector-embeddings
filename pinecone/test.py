from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
loader =  OnlinePDFLoader("")
data = loader.load()
print(data)
