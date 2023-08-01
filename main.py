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
# expose this index in a retriever interface
print("loading embedding")
# Embeddings
embeddings = HuggingFaceEmbeddings()
print("embedding loaded")
db= FAISS.load_local("faiss_index", embeddings)
query = "what is the red horse about"

docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores[0][0])
print(docs_and_scores[0][1])

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
# create a chain to answer questions
print("downloading LLM model")
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1024})
print("loaded llm model")

print("loading QA")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
print("QA loaded")

result = qa({"query": query})
print(result)
