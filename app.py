import streamlit as st
from models.predict_model import Predict
import traceback

# Initialize the RAG model
predict = Predict()

# Define the template
template = """
    انت مساعد ذكي للاشخاص الناطقين باللغة العربيه تساعدهم في معرفة اخر الاخبار في مجالات مثل التكنولوجيا والسياسة والرياضة من الملفات المتاح لك الاطلاع عليها ,وفي نهاية اجابتك اشكر المستخدم واسئله هل لديك اسئلة اخري 
    
knowledge you know:
{context}

Question: {question}

ماذا تفعل إذا لم تكن الإجابة مدرجة في السؤال أو السياق:
1. أخبر المستخدم أنك ليس لديك معلومات كافية للاجابة علي سواله
2. أخبر المستخدم أنك متخصص في فئات المقالات [الثقافة، والمال، والطب، والسياسة، والدين، والرياضة، والتكنولوجيا] فقط من المواقع الإلكترونية: الخليج، والعربية، وأخبارنا..
3. اسأل المستخدم إذا كان لديه المزيد من الأسئلة ليطرحها.
4. لا تذكر أي شيء عن السياق.

answer:
"""

# question = 'من هو رئيس مصر؟'
# ans = predict.get_answer(template, question)

# print(ans)

import logging

# Streamlit UI
st.title("RAG-Powered Arabic AI Assistant")

question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if question:
        logging.info("Question submitted: %s", question)
        print(question)
        try:
            answer = predict.get_answer(template, question)
            logging.info("Answer generated successfully.")
            st.markdown(answer, unsafe_allow_html=True)
        except Exception as e:
            logging.error("Error occurred while generating the answer: %s", e)
            st.error("An error occurred while generating the answer. Please try again later.")
    else:
        st.error("Please enter a question.")
