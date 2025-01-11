# rag_system.py

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the generator mode (using GPT-2 or other transformer models)
generator = pipeline('text-generation', model='gpt2')

# Initialize the retriever (using Sentence-BERT for encoding documents)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to initialize the FAISS index for document retrieval
def create_faiss_index(documents):
    embeddings = embedder.encode(documents, convert_to_tensor=True)
    embeddings = np.array(embeddings)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1]) # L2 distance for cosine similarity
    index.add(embeddings)

    return index

# Function to retrieve the top-k most similar documents based on query
def retrieve_documents(query, index, documents, top_k=3):
    query_embedding = embedder.encode([query])[0]
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Return the top-k most similar documents
    retrieved_docs = [documents[i] for i in indices [0]]
    return retrieved_docs

# Function to generate text based on retrieved documents
def generate_text_from_documents(query, retrieved_docs):
    # Combine the retrieved documents into a single context
    context = " ".join(retrieved_docs)

    # Generate text using the GPT-2 model
    input_text = f"Query: {query}\nContext: {context}\nResponse:"
    generated = generator(input_text, max_length=150, num_return_sequences=1)

    return generated[0]['generated_text']

# Example usage
if __name__ == "__main__":
    # Example documents (these can be loaded from your data folder)
    documents = [
            ""
            ""
            ""
        ]

    # Create FAISS index
    index = create_faiss_index(documents)

    # Example query
    query = "What is Machine Learning?"

    # Retrieve documents
    retrieved_docs = retrieve_documents(query, index, documents)
    print(f"Retrieved Documents: {retrieved_docs}")

    # Generate a response based on the retrieved documents
    response = generate_text_from_documents(query, retrieved_docs)
    print(f"Generated Response: {response}")

