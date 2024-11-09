import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import faiss
import pickle
import os
from typing import List, Tuple
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
env_path = os.path.join(os.getcwd(), '.env')
# client = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FAQEmbeddingSystem:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.dimension = 1536  # Dimension of OpenAI embeddings
        self.index = None
        self.faqs = None
        
    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for a text"""
        response = client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, df: pd.DataFrame, force_rebuild: bool = False):
        """Build or load FAISS index"""
        index_file = 'faiss_index.pkl'
        embeddings_file = 'faq_embeddings.pkl'
        
        if not force_rebuild and os.path.exists(index_file) and os.path.exists(embeddings_file):
            # Load existing index and FAQs
            with open(index_file, 'rb') as f:
                self.index = pickle.load(f)
            with open(embeddings_file, 'rb') as f:
                self.faqs = pickle.load(f)
        else:
            # Create combined text for embeddings
            texts = [
                f"Question: {row['Question']}\nAnswer: {row['Answer']}"
                for _, row in df.iterrows()
            ]
            
            # Get embeddings
            embeddings = self.create_embeddings(texts)
            
            # Build FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            
            # Store FAQs
            self.faqs = df
            
            # Save index and FAQs
            with open(index_file, 'wb') as f:
                pickle.dump(self.index, f)
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.faqs, f)

    def find_similar_questions(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """Find k most similar questions"""
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return list(zip(indices[0], distances[0]))

def load_faq_data(file_path='faq_data.csv'):
    """Load FAQ data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("FAQ data file not found. Please check the file path.")
        return None

def get_openai_response(question: str, context_faqs: pd.DataFrame):
    """Get response from OpenAI API using relevant FAQ context"""
    # Convert relevant FAQs to context string
    context = "\n".join([
        f"Question: {row['Question']}\nAnswer: {row['Answer']}"
        for _, row in context_faqs.iterrows()
    ])
    
    prompt = f"""You are a helpful assistant for CodeBasics, an e-learning platform that 
    teaches data-related courses and bootcamps. Use the following most relevant FAQ entries 
    to answer the student's question. If the answer isn't fully covered in the provided FAQs, 
    you can provide additional helpful information based on common educational practices, 
    but clearly indicate which parts are general advice versus FAQ-based information.

    Relevant FAQ Entries:
    {context}

    Student Question: {question}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant for CodeBasics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'embedding_system' not in st.session_state:
        st.session_state.embedding_system = None
    if 'faq_df' not in st.session_state:
        st.session_state.faq_df = None

def main():
    st.set_page_config(
        page_title="CodeBasics Q&A System",
        page_icon="🎓",
        layout="centered"
    )

    # Initialize session state
    initialize_session_state()

    # Add header and description
    st.title("🎓 CodeBasics Q&A Assistant")
    st.markdown("""
    Welcome to the CodeBasics Q&A system! Ask any questions about our courses, 
    bootcamps, or general learning-related queries.
    """)

    # Load and process FAQ data if not already done
    if st.session_state.embedding_system is None:
        with st.spinner('Loading FAQ database...'):
            st.session_state.faq_df = load_faq_data()
            if st.session_state.faq_df is not None:
                st.session_state.embedding_system = FAQEmbeddingSystem()
                st.session_state.embedding_system.build_index(st.session_state.faq_df)

    if st.session_state.embedding_system is not None:
        # Create the text input for user questions
        user_question = st.text_input(
            "Ask your question here:",
            placeholder="e.g., What courses do you offer for data science?"
        )

        # Add a submit button
        if st.button("Get Answer", type="primary"):
            if user_question:
                with st.spinner('Finding the best answer for you...'):
                    # Find similar questions
                    similar_questions = st.session_state.embedding_system.find_similar_questions(user_question)
                    
                    # Get relevant FAQ entries
                    relevant_indices = [idx for idx, _ in similar_questions]
                    relevant_faqs = st.session_state.faq_df.iloc[relevant_indices]
                    
                    # Get and display the response
                    response = get_openai_response(user_question, relevant_faqs)
                    st.markdown("### Answer:")
                    st.markdown(response)
                    
                    # Show similar questions (optional)
                    with st.expander("View Similar FAQ Questions"):
                        for idx, distance in similar_questions:
                            st.markdown(f"- {st.session_state.faq_df.iloc[idx]['Question']}")
                    
                    # Add feedback section
                    st.markdown("---")
                    st.markdown("### Was this answer helpful?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Yes"):
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 No"):
                            st.text_area("Please tell us how we can improve:", key="feedback")
                            if st.button("Submit Feedback"):
                                st.success("Thank you for your feedback! We'll work on improving.")
            else:
                st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()
