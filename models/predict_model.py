# rag_system.py

import os
import pickle
import markdown
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Directories
SAMPLE_DIRECTORY = r'D:\Projects\RAG-Powered-Arabic-AI-Assistant\data\processed\sys_sample.pkl'
PERSIST_DIRECTORY = r'D:\Projects\RAG-Powered-Arabic-AI-Assistant\data\processed\sample_vector_db2'

# Retrieve the Google API Key from environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')

if not google_api_key:
    raise ValueError("Did not find google_api_key, please add an environment variable `GOOGLE_API_KEY` which contains it, or pass `google_api_key` as a named parameter.")

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
    
    def embed_documents(self, docs):
        embeddings = self.model.encode(docs)
        return embeddings.tolist()
    
    def embed_query(self, query):
        return self.model.encode(query).tolist()

class Predict:
    def __init__(self):
        
        embed_model = Embedding()
        self.vector_data = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embed_model
        )
        self.retriever = self.vector_data.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key
        )

    def get_chunk_by_index(self, index):
        with open(SAMPLE_DIRECTORY, 'rb') as f:
            chunks = pickle.load(f)
        return chunks[index]
    
    def test_retriever(self, text):
        return self.retriever.invoke(text)
    
    def get_answer(self, template, question):
        try:
            print("Generating answer...")
            custom_rag_prompt = ChatPromptTemplate.from_template(template)
            rag_chain = (
                {'context': self.retriever, 'question': RunnablePassthrough()}
                | custom_rag_prompt
                | self.llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)
            print("Answer generated:", answer)
            return answer
        except Exception as e:
            print("Error in get_answer:", e)
            return "An error occurred while generating the answer."
