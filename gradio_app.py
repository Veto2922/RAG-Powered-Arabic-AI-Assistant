# app.py
from models.predict_model import Predict
import os
import warnings
import gradio as gr
import pandas as pd

warnings.filterwarnings("ignore")

# Retrieve the Google API key from environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')

if not google_api_key:
    raise ValueError("Did not find google_api_key, please add an environment variable `GOOGLE_API_KEY` which contains it, or pass `google_api_key` as a named parameter.")

# Initialize the RAG model
predict = Predict(google_api_key)
predict.get_chunk_by_index(35000)

def answer_and_retrieve(question):
    """Function to handle the Gradio interface, providing both answer and list of similar documents."""
    try:
        # Get the answer to the question
        answer = predict.get_answer(question)
        
        # Retrieve the most similar documents based on the question
        similar_docs = predict.test_retriever(question)  # Assume this returns a list of similar documents
        
        # Check if similar_docs is a list of strings
        if not isinstance(similar_docs, list):
            raise ValueError("Expected a list of documents.")

        # Convert each document to a string if needed
        similar_docs = [str(doc) for doc in similar_docs]

        # Format documents into a DataFrame
        docs_df = pd.DataFrame(similar_docs, columns=["Similar Documents"])
        
        return answer, docs_df
    except Exception as e:
        return str(e), pd.DataFrame(columns=["Similar Documents"])


# Create the Gradio interface
iface = gr.Interface(
    fn=answer_and_retrieve,
    inputs=gr.Textbox(label="Ask a Question", placeholder="Enter your question here..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Dataframe(label="Most Similar Documents", headers=["Similar Documents"], row_count=5)  # Adjust row_count as needed
    ],
    title="RAG System",
    description="Ask a question to get an answer and see the most similar documents."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
