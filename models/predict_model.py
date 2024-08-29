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
from .Check_files_exist import GoogleDriveDownloader

# Load environment variables from the .env file
load_dotenv()

# Initialize the downloader class
downloader = GoogleDriveDownloader()

# Download the sample file and persist folder
downloader.download_sample_file()
downloader.download_persist_folder()
# Directories
SAMPLE_DIRECTORY = r'data\processed\all_split.pkl'
PERSIST_DIRECTORY = r'data\processed\vector_db'

# Retrieve the Google API Key from environment variables

# Define the template
template = """
    انت مساعد ذكي للاشخاص الناطقين باللغة العربيه تساعدهم في معرفة اخر الاخبار في مجالات مثل التكنولوجيا والسياسة والرياضة من الملفات المتاح لك الاطلاع عليها او البيانات التي تدربت عليها مسبقا ,وفي نهاية اجابتك اشكر المستخدم واسئله هل لديك اسئلة اخري 
    
knowledge you know:
{context}

Question: {question}

ماذا تفعل إذا لم تكن الإجابة مدرجة في السؤال أو السياق او في البيانات التي تدرب عليها مسبقا:
1. أخبر المستخدم أنك ليس لديك معلومات كافية للاجابة علي سواله
3. اسأل المستخدم إذا كان لديه المزيد من الأسئلة ليطرحها بطريقة لطيفة ولائقة.
4. لا تذكر أي شيء عن السياق.

answer:
"""

class Embedding:
    def __init__(self , model):
        self.model = model
    
    def embed_documents(self, docs):
        embeddings = self.model.encode(docs)
        return embeddings.tolist()
    
    def embed_query(self, query):
        return self.model.encode(query).tolist()


# ------------------------------------------------------------------------------------

class Predict:
    def __init__(self , google_api_key):
        
        model =  SentenceTransformer('all-distilroberta-v1')
        embed_model = Embedding(model)
        
        self.vector_data = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embed_model
        )
        self.retriever = self.vector_data.as_retriever(search_type='similarity', search_kwargs={'k': 12})
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key
        )

    def get_chunk_by_index(self, index):
        with open(SAMPLE_DIRECTORY, 'rb') as f:
            chunks = pickle.load(f)
        return chunks[index]
    
    def test_retriever(self, text):
        return self.retriever.invoke(text)
    
    def get_answer(self, question):
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
            print("Answer generated:")
            return answer
        except Exception as e:
            print("Error in get_answer:", e)
            return "An error occurred while generating the answer please check your API."


# pred =  Predict()


# text = 'من هو حاكم السعودية'
# templot = 'plese answer the following what is the football'
# print(pred.get_answer(templot , text))