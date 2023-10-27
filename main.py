
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset
cloud_config = {
    'secure_connect_bundle':ASTRA_DB_SECURE_BUNDLE_PATH
}

auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID,ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud = cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key = OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
myCassandraVstore = Cassandra(embedding = myEmbedding,
                              session = astraSession,
                              keyspace = ASTRA_DB_KEYSPACE,
                              table_name="qa_mini_demo",)

print("loading data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\ngenerating embeddings and storing in astradb")
myCassandraVstore.add_texts(headlines)

print("inserted %i headlines.\n" % len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVstore)
first_question = True
while True:
    if first_question:
        query_text = input("\nenter your question: ")
        first_question = False
    else:
        query_text = input("\nwhat's your next question: ")
    if query_text.lower() == "quit":
        break
    print("question:\"%s\"" % query_text)
    answer = vectorIndex.query(query_text,llm=llm).strip()
    print("Answer: \"%s\"\n" % answer)

    print("Documents by relevance")
    for doc,score in myCassandraVstore.similarity_search_with_score(query_text,k=4):
        print(" %0.4f \"%s ...\"" % (score,doc.page_content[:60]))
