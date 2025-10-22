import gradio as gr
import os
import getpass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pdfingestor import PDFIngestor
from langchain_core.documents import Document
from chromadb.config import Settings
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from threading import Thread
import time

# Initialize LLM and embedding functions
conversation_model = OllamaLLM(model="qwen3:14b", temperature=0.3)
ollama_embedding_function = OllamaEmbeddingFunction(model_name="nomic-embed-text")
db = chromadb.Client()
vector_store = db.create_collection("vectorstore", get_or_create=True, embedding_function=ollama_embedding_function)

# Global variables to store loaded documents
loaded_chunks = []
loaded_filename = ""
conversation_history = []

# Prompt template
prompt = """
               You are a helpful AI roleplaying game expert system that answers questions based on the provided context.
               Rules:
               1. Only use information from the provided context to answer questions except when asked to perform tasks specifically requiring creativity such as generating adventures or character biographies.
               2. If the context doesn't contain enough information, say so honestly
               3. Be specific and cite relevant parts of the context
               4. Keep your answers clear and concise
               5. If you're unsure, admit it rather than guessing
               Context:
               {context}
               Question: {input}

               Answer based on the context above:
               """
prompt_template = ChatPromptTemplate.from_template(prompt)

def build_vector_store(chunks):
    """Build vector store from chunks in a separate thread"""
    global loaded_chunks, vector_store
    ids = [f"{i}" for i in range(len(chunks))]
    vector_store.add(documents=chunks, ids=ids)
    loaded_chunks = chunks
    print("docs added to vector store!")

def ingest_book_pdf(file_path):
    """Process PDF file and build vector store"""
    global loaded_filename, conversation_history
    try:
        print("ingesting book pdf")
        loaded_filename = os.path.basename(file_path)
        conversation_history = []  # Reset conversation history when loading new PDF
        content = PDFIngestor.read_pdf_file(file_path)
        content = PDFIngestor.clean_extracted_text(content)
        chunks = PDFIngestor.chunk_text(content)
        print(f"Adding {len(chunks)} chunks")
        
        # Build vector store in a separate thread
        task_thread = Thread(target=build_vector_store, args=(chunks,))
        task_thread.daemon = True
        task_thread.start()
        
        return f"Successfully loaded {loaded_filename} with {len(chunks)} chunks"
    except Exception as e:
        return f"Failed to load PDF: {str(e)}"

def get_answer(question):
    """Get answer to question using RAG"""
    if not loaded_chunks:
        return [("assistant", "Please load a PDF file first.")]
    
    try:
        query_vec = ollama_embedding_function([question])
        result = vector_store.query(query_embeddings=query_vec, n_results=5)
        context = result["documents"][0]
        
        rag_chain = (prompt_template | conversation_model | StrOutputParser())
        response = rag_chain.invoke({"context": context, "input": question})
        
        # Add to conversation history
        conversation_history.append(("user", question))
        conversation_history.append(("assistant", response))
        
        return conversation_history
    except Exception as e:
        error_response = f"Failed to get answer: {str(e)}"
        conversation_history.append(("user", question))
        conversation_history.append(("assistant", error_response))
        return conversation_history

def update_ui_state():
    """Update UI state based on whether chunks are loaded"""
    return gr.update(interactive=not not loaded_chunks)

# Create Gradio interface
with gr.Blocks(title="Roleplaying Agent", theme=gr.themes.Default()) as demo:
    gr.Markdown("# Roleplaying Agent")
    gr.Markdown("Upload a PDF book to ask questions about it.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF Book", file_types=[".pdf"])
            ingest_button = gr.Button("Load Book")
            status_text = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            # Single chat interface
            chatbot = gr.Chatbot(label="Chat with PDF", height=500)
            msg = gr.Textbox(label="Your Message", placeholder="Ask a question about the loaded book...", interactive=False)
            clear = gr.Button("Clear Chat")
    
    # Event handling
    ingest_button.click(
        fn=ingest_book_pdf,
        inputs=pdf_input,
        outputs=status_text
    ).then(
        fn=update_ui_state,
        inputs=None,
        outputs=msg
    )
    
    # Handle both Enter key and button click for sending messages
    msg.submit(
        fn=get_answer,
        inputs=msg,
        outputs=chatbot
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=msg
    )
    
    clear.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[chatbot, msg]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

