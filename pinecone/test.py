from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
loader =  OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
data = loader.load()
print(data)