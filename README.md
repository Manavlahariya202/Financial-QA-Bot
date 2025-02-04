
1.  This project is a financial term query system that uses Llama 2 and Ollama embeddings for efficient processing on low-end systems.
Note: I opted for Ollama Embeddings and Llama 2 due to system memory constraints instead of using the resource-intensive Llama 3.2 model.

2.  Installation:

Ensure you have Python installed on your system.
Install all the required libraries using the following command:
pip install -r requirements.txt

3. Running the Application:

Start the Streamlit app by running the following command:
streamlit run app.py
4.  Functionality:
Upload a financial PDF through the app interface.
Query for financial terms or insights directly.

5. Demo:

Refer to the attached demo images for a preview of the app's functionality and interface.
Notes:

The app is optimized to work on systems with limited resources.
It uses FAISS for efficient vector retrieval and RecursiveCharacterTextSplitter for document chunking.
