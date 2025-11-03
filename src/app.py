import os
import glob
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[dict]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents with content and metadata
    """
    results = []
    
    # Define the data directory
    data_dir = r"C:\Users\DELL\Desktop\week 4 project\Project-1\data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} directory not found. Creating sample documents...")
        # Return some sample documents if no data directory exists
        return None
    
    # Find all markdown files in the data directory
    pattern = os.path.join(data_dir, "**", "*.md")
    markdown_files = glob.glob(pattern, recursive=True)
    
    # Also look for .txt files
    txt_pattern = os.path.join(data_dir, "**", "*.txt")
    txt_files = glob.glob(txt_pattern, recursive=True)
    
    all_files = markdown_files + txt_files
    
    if not all_files:
        print(f"No documents found in {data_dir}. Using sample documents...")
        return [
            {
                'content': 'Python is a high-level programming language known for its simplicity and readability.',
                'metadata': {'source': 'sample_python.txt', 'topic': 'programming'}
            },
            {
                'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
                'metadata': {'source': 'sample_ml.txt', 'topic': 'AI'}
            }
        ]
    
    print(f"Found {len(all_files)} documents")
    
    # Read each file
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                
                results.append({
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'filename': os.path.basename(file_path)
                    }
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
you must obey the following rules:
1. DO not answer any question out side the documents below.
2. Answer questions in a clear and polite manner,never use ambiguous words for you response
3. If you are asked any question outside this document kindly say"I do not have such data with me.
4. Make your response less than 80 words
5. Never Hallucinate

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt_template = ChatPromptTemplate.from_template(template)

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's question
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM
        """
        # Step 1: Retrieve relevant context from vector database
        search_results = self.vector_db.search(input, n_results=n_results)
        
        # Step 2: Combine retrieved chunks into a single context string
        retrieved_docs = search_results.get('documents', [])
        
        if not retrieved_docs:
            return "I couldn't find any relevant information in the knowledge base to answer your question."
        
        # Join all retrieved documents with separators
        context = "\n\n---\n\n".join(retrieved_docs)
        
        # Step 3: Generate answer using the LLM chain
        llm_answer = self.chain.invoke({
            "context": context,
            "question": input
        })
        
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        # Add documents to the assistant
        assistant.add_documents(sample_docs)

        print("\n" + "="*50)
        print("RAG Assistant is ready! Ask me anything.")
        print("Type 'quit' or 'exit' to stop.")
        print("="*50 + "\n")

        done = False

        while not done:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                done = True
                print("\nGoodbye!")
            elif not question:
                print("Please enter a question.")
            else:
                print("\nThinking...\n")
                result = assistant.query(question)
                print(f"Answer: {result}")

    except Exception as e:
        print(f"\nError running RAG assistant: {e}")
        print("\nMake sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()