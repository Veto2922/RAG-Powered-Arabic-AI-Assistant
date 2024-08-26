

# RAG-Powered-Arabic-AI-Assistant

This project is an AI-powered QA assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant answers to customer questions in Arabic. The assistant retrieves relevant information from the SANAD dataset, combines it with generative AI capabilities, and delivers precise and detailed responses across various domains like technology, politics, sports, and more.

## To use this project

1. **Python Version**: Ensure you are using Python 3.10 or above.
2. **Requirements**: Install the required packages by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Required Files**:
    - [File 1](https://drive.google.com/file/d/1ibbCH69F0W_e7xKjCRX1PW3cvPJIA-Fk/view?usp=sharing)
    - [File 2](https://drive.google.com/drive/folders/1kCUYhA0uxuk6PKBHhoyClPzXvj6_62qr?usp=sharing)

4. **open models folder then modified predict_model.py in files path you download**

5. **Set Up Environment Variables**:
    - Create a `.env` file and add your `GOOGLE_API_KEY` and `AI2_API_KEY`.
    
    Example `.env` file:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    AI2_API_KEY=your_ai2_api_key
    ```
6. **Open test_pred notebook and start the system (don't use app.py or flask_app.py it is still under development)**

## Project Structure

- **Vector DB Notebook**: Handles the creation of vector databases using embeddings from text files across various domains. The Chroma vector store is used to store and retrieve embeddings.
- **RAG Model Notebook**: Implements the RAG system, integrating a language model with retrieval capabilities to generate contextually relevant responses.



## Data source:
https://www.kaggle.com/datasets/haithemhermessi/sanad-dataset/data

## Notes

- The project supports multiple domains, including Technology, Culture, Finance, Medical, Politics, Religion, and Sports.
- The system uses systematic sampling to reduce the data size for efficient processing.
- Several models are integrated, including `gemini-pro`, `jamba-1.5-mini`, and `facebook/bart-large`, offering flexibility and performance tuning.
- Future improvements include fully developing and testing the `app.py` and `flask_app.py` files for a production-ready application.

## Acknowledgments

- This project is based on a Cookiecutter Data Science template.
- The SANAD dataset is used for training and testing.

---

This updated README provides a more comprehensive overview of the project, reflecting the detailed workflows and methods described in the notebooks.
