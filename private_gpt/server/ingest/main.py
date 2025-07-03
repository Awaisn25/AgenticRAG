import fitz  # PyMuPDF for PDF text extraction
import uuid  # For generating UUIDs
from qdrant_client import QdrantClient
from llama_index.core import Document
from transformers import AutoTokenizer, AutoModel  # For Hugging Face embeddings
import torch  # PyTorch for model handling
import ollama  # Ollama for offline LLM-based question answering
import logging  # To add logging
import numpy as np  # For generating additional arrays

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Function to check if the collection exists, and create it if not
def create_collection_if_not_exists(collection_name, vector_size=1024, distance_metric="Cosine"):
    try:
        qdrant_client.get_collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.info(f"Collection '{collection_name}' not found. Creating it now...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": vector_size,
                "distance": distance_metric
            }
        )
        logging.info(f"Collection '{collection_name}' created successfully.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        if not page_text.strip():
            logging.warning(f"Empty text found on page {page.number}")
        text += page_text
    if not text.strip():
        logging.warning(f"No text extracted from the PDF.")
    return text

# Function to split text into chunks of given size
def split_text_into_chunks(text, chunk_size=1000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Create a custom Qdrant Vector Store to interact with Qdrant
class EmbeddingComponent:
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name

    def insert(self, embeddings, ids, texts, arrays=None):
        points = []
        for id, embedding, text in zip(ids, embeddings, texts):
            payload = {"text": text}
            if arrays is not None:
                payload["array_data"] = arrays  # Include the additional label-value pair data in the payload
            
            points.append({"id": id, "vector": embedding, "payload": payload})

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def query(self, query_vector, top_k=5):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

# Load Hugging Face model and tokenizer
model_path = "models/bge-m3-model"  # Adjust model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def _get_text_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    if not embedding:
        logging.warning(f"Empty embedding for text: {text[:50]}...")
        return None    
    return embedding

def _get_query_embedding(query: str):
    return _get_text_embedding(query)

# Store embeddings in Qdrant along with additional label-value pair data
def store_embeddings_in_qdrant(text_chunks, additional_array_data=None):
    vector_store = EmbeddingComponent(client=qdrant_client, collection_name="pdf_embeddings")    
    documents = []
    for chunk in text_chunks:
        embedding = _get_text_embedding(chunk)
        if embedding is None:
            logging.warning(f"Skipping chunk due to empty embedding: {chunk[:50]}...")
            continue
        doc = Document(text=chunk, embedding=embedding)
        documents.append(doc)
    
    embeddings = [doc.embedding for doc in documents]
    texts = [doc.text for doc in documents]
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    
    # If additional label-value pair data is provided, use it, otherwise default to None
    arrays = additional_array_data if additional_array_data is not None else None
    
    if embeddings:
        vector_store.insert(embeddings, ids, texts, arrays)
        logging.info(f"Inserting {len(embeddings)} embeddings and texts into Qdrant...")
    else:
        logging.warning("No valid embeddings to insert.")

# Query Qdrant for relevant text chunks
def query_qdrant(query_text):
    vector_store = EmbeddingComponent(client=qdrant_client, collection_name="pdf_embeddings")    
    query_embedding = _get_query_embedding(query_text)    
    if query_embedding is None:
        logging.warning("Query resulted in empty embedding. Skipping query.")
        return    
    results = vector_store.query(query_embedding, top_k=5)
    return results

# Ollama model for context-based question answering
def ollama_query(context, query):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
        content = response.get('message', {}).get('content', 'No content available')
        return content
    except Exception as e:
        logging.error(f"Error querying Ollama: {e}")
        return "Error occurred while querying Ollama."

# Main function to process PDF and insert into Qdrant
def process_pdf_and_insert(pdf_path, additional_array_data=None):
    create_collection_if_not_exists(collection_name="pdf_embeddings", vector_size=1024, distance_metric="Cosine")   
    text = extract_text_from_pdf(pdf_path)   
    text_chunks = split_text_into_chunks(text)    
    store_embeddings_in_qdrant(text_chunks, additional_array_data)

# Querying with context-based answer generation using Ollama
def query_with_context(query_text):
    results = query_qdrant(query_text)
    if not results:
        logging.info("No relevant results found in Qdrant.")
        return

    # Get the top result and generate an answer using Ollama
    top_result = results[0]
    context = top_result.payload['text']
    answer = ollama_query(context, query_text)
    return answer

# Example usage
# pdf_path = 'data/book1.pdf'  # Replace with the actual path to your PDF file

# Example of additional label-value pair data
additional_array_data = {
    "topic": "Science",
    "length": 5,  # This could be the length of the text chunk or any other metadata
    "importance": 8.5  # A score for the importance of this chunk
}

process_pdf_and_insert(pdf_path="data/book1.pdf", additional_array_data=additional_array_data)

query_text = "Provide summary of Prophet Muhammad (pbuh) marriage in 20 lines"
answer = query_with_context(query_text)
print("Answer:", answer)
