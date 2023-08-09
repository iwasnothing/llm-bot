
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.redis import Redis
import textwrap
import nltk
import ssl
import os

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
pdf_folder_path=""
loader = PyPDFDirectoryLoader(pdf_folder_path)
documents = loader.load()


print("split text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
print("document splitted")
print("loading embedding")
# Embeddings
embeddings = HuggingFaceEmbeddings()
#embeddings = VertexAIEmbeddings()

print("embedding loaded")
# select which embeddings we want to use
# create the vectorestore to use as the index
dbhost=""
user=""
url="redis://" + user + "@" + dbhost
db = Redis.from_documents(
    texts, embeddings, redis_url=url, index_name="link"
)
print("vector db loaded")
query = "what is the red horse about"
context_docs = db.similarity_search(query)
print(context_docs)

docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores)

