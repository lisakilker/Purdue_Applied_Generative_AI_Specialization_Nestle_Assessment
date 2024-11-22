import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
import os
from string import Template
import gradio as gr
from striprtf.striprtf import rtf_to_text

#Load OpenAI API Key from .rtf file
#Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
#Path to key.rtf file
key_file_path = os.path.join(current_dir, "key.rtf")

#Read the API key from the .rtf file
try:
    with open(key_file_path, "r") as file:
        raw_content = file.read()
        #Extract plain text from RTF
        openai_api_key = rtf_to_text(raw_content).strip()
except FileNotFoundError:
    raise FileNotFoundError(f"API key file not found at: {key_file_path}")
except ValueError as e:
    raise ValueError(f"Error reading API key: {e}")

#Set OpenAI API Key
openai.api_key = openai_api_key

#Initialize ChromaDB client
client = chromadb.Client()

#Define the prompt template
prompt_template = Template("""
You are an AI assistant specializing in answering questions based on provided context from a document. Your task is to use the context to answer questions accurately and concisely. If the context does not contain enough information to answer the question, say, "The provided context does not contain sufficient information to answer this question."

Context:
${context}

Question:
${question}

Answer concisely and clearly:
""")

#Function to generate a prompt using the template
def generate_prompt(context, question):
    return prompt_template.substitute(context=context, question=question)

#Step 1: Load and Split PDF
def process_pdf(pdf_name):
    #Locate PDF file in the same folder
    pdf_path = os.path.join(current_dir, pdf_name)
    try:
        #Load the PDF
        loader = PyPDFLoader(pdf_path)  
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load the PDF: {e}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    #Use new ChromaDB client configuration
    collection = client.get_or_create_collection("pdf_chunks")

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i+1}"
        chunk_content = chunk.page_content

        print(f"Processing chunk {i+1}/{len(chunks)}...")
        collection.add(
            documents=[chunk_content],
            metadatas=[{"chunk_id": chunk_id}],
            ids=[chunk_id]
        )
    print("All chunks have been processed and stored in ChromaDB.")

#Step 2: Question-Answering System
def ask_question(query, collection_name="pdf_chunks", n_results=3):
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant context found to answer your question."

    context = "\n".join(results["documents"][0])
    prompt = generate_prompt(context=context, question=query)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    answer = response["choices"][0]["message"]["content"].strip()
    return answer

#Gradio Interface
def chatbot_interface(user_question):
    try:
        answer = ask_question(user_question)
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

#Process the PDF (only needed once; comment out after first run)
pdf_name = "nestle.pdf"
process_pdf(pdf_name)

#Build Gradio Interface
gr_interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="Enter your question:", placeholder="e.g., What is Nestle's shared responsibility?"),
    outputs=gr.Textbox(label="Answer:"),
    title="Nestle's AI Assistant",
    description="Ask questions based on the content of the uploaded PDF."
)

#Launch the Interface
if __name__ == "__main__":
    gr_interface.launch()