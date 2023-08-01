# Document Loader
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
import textwrap
import nltk
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')

print("loading doc")
path = "data"
loader = DirectoryLoader(path)
documents = loader.load()
print("document loaded")
print("split text")
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
print("document splitted")
print("loading embedding")
# Embeddings
embeddings = HuggingFaceEmbeddings()
print("embedding loaded")
# select which embeddings we want to use
# create the vectorestore to use as the index
#db = Chroma.from_documents(texts, embeddings)
db = FAISS.from_documents(texts, embeddings)
print("vector db loaded")
query = "what is the red horse about"
context_docs = db.similarity_search(query)
print(context_docs)

docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores)

db.save_local("faiss_index")