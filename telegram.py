import telegram
from langchain import PromptTemplate
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from langchain import HuggingFaceHub
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
import textwrap
import nltk
import ssl
import os
from transformers import AutoModel, AutoConfig



# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
# expose this index in a retriever interface
print("loading embedding")
# Embeddings
embeddings = HuggingFaceEmbeddings()
print("embedding loaded")
dbhost=""
user=""
url="redis://" + user + "@" + dbhost
db = Redis.from_existing_index(
    embeddings, redis_url=url, index_name="link"
)


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
# create a chain to answer questions
print("downloading LLM model")

max_input_length = 5000

print(max_input_length)
# 512
llm = VertexAI()
print("loaded llm model")

print("loading QA")
chain = load_qa_chain(llm, chain_type="stuff")
print("QA loaded")


template = """Answer the question in complete sentences and under 200 words based on the context below. 

Context: {context}

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query","context"],
    template=template
)


TOKEN = ''

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text='Hello! I am your Telegram chatbot about banking regulations.')

def handle_message(update, context):
    text = update.message.text
    if text == '/greet':
        context.bot.send_message(chat_id=update.message.chat_id, text='Hello!')
    else:
        #docs = db.similarity_search(text)
        docs = db.similarity_search_with_score(text)
        filtered = []
        for doc, _score in docs:
            #print(f"{_score} - {doc.page_content}")
            if _score < 0.75:
                filtered.append(doc)
        if len(filtered) > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_input_length, chunk_overlap=1)
            context_texts = text_splitter.split_documents(filtered)
            if len(text) >= max_input_length:
                query = text[:max_input_length]
            else:
                query = text
            #context_docs = []
            total_len = 0
            sources = []
            contents = []
            for doc in context_texts:
                file = os.path.basename(doc.metadata['source'])
                if file not in sources:
                    sources.append(file)
                total_len = total_len + len(doc.page_content)
                if total_len < max_input_length:
                    contents.append(doc.page_content.replace("\n", " "))

            #result = chain.run(input_documents=filtered, question=text, return_only_outputs=True)
            background = "\n".join(contents)
            result = llm(prompt_template.format(query=query,context=background))
            print(background)
            print(query)
            print(result)
            context.bot.send_message(chat_id=update.message.chat_id, text="The answer is: "+result)
            context.bot.send_message(chat_id=update.message.chat_id, text="the infomation comes from: ")
            for src in set(sources):
                uri="https://cdn.amcm.gov.mo/uploads/files/banking_sector/rules_and_guideline/notices_and_guidelines/"
                context.bot.send_message(chat_id=update.message.chat_id, text=uri+src)
            context.bot.send_message(chat_id=update.message.chat_id, text="the context comes from: ")
            context.bot.send_message(chat_id=update.message.chat_id, text=background[:256] + "  ......")
        else:
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text="sorry, I can't get you")
            #context.bot.send_message(chat_id=update.message.chat_id, text=context_docs[0].page_content)


updater = Updater(TOKEN, use_context=True)

dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))
dp.add_handler(MessageHandler(Filters.text, handle_message))

updater.start_polling()
updater.idle()
