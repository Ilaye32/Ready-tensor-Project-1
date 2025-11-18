# RAG Assistant 🤖

A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over your document collection. This project demonstrates a complete RAG workflow using vector databases and multiple LLM providers.

## 🌟 Features

- **Multi-LLM Support**: Compatible with OpenAI GPT, Groq Llama, and Google Gemini models
- **Vector-Based Search**: Efficient document retrieval using ChromaDB
- **Document Processing**: Automatically loads and processes Markdown and text files
- **Interactive CLI**: User-friendly command-line interface for queries
- **Configurable Retrieval**: Adjustable number of relevant chunks per query
- **Environment-Based Configuration**: Secure API key management via `.env` files

## 📋 Prerequisites

- Python 3.8 or higher
- At least one API key from:
  - OpenAI
  - Groq
  - Google AI (Gemini)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```env
# Choose at least one LLM provider

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Groq (Optional)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini (Optional)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.0-flash
```

## 📁 Project Structure

```
.
├── data/                          # Your document collection
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   └── ...
├── src/
│   ├── __pycache__/
│   ├── chroma_db/                 # Vector database storage
│   ├── app.py                     # Main RAG application
│   ├                 # Path configurations
│   └── vectordb.py                # Vector database implementation
├── .env                           # Environment variables (create this)
├── .env.example                   # Example environment file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 💻 Usage

1. **Add your documents**

Place your `.txt` or `.md` files in the `data/` directory. The system will automatically:
- Scan for all text and markdown files
- Load and process them
- Create vector embeddings
- Store them in ChromaDB

2. **Run the assistant**

```bash
python src/app.py
```

3. **Ask questions**

```
RAG Assistant is ready! Ask me anything.
Type 'quit' or 'exit' to stop.
==================================================

Your question: What is artificial intelligence?

Thinking...

Answer: Artificial intelligence (AI) refers to the simulation of human 
intelligence processes by machines, particularly computer systems...
```

## 🛠️ Configuration

### Adjusting Retrieval Parameters

In `app.py`, modify the `query()` method's `n_results` parameter:

```python
result = assistant.query(question, n_results=5)  # Retrieve top 5 chunks
```

### Customizing the Prompt

Edit the prompt template in `RAGAssistant.__init__()` to change the assistant's behavior:

```python
template = """You are a helpful AI assistant. Use the following context...
[Your custom instructions here]
"""
```

## 📦 Dependencies

- `langchain` - LLM orchestration framework
- `langchain-openai` - OpenAI integration
- `langchain-groq` - Groq integration
- `langchain-google-genai` - Google Gemini integration
- `chromadb` - Vector database
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client
- Additional dependencies in `requirements.txt`

## 🔧 How It Works

1. **Document Loading**: The system scans the `data/` directory for `.txt` and `.md` files
2. **Vectorization**: Documents are converted into vector embeddings using ChromaDB
3. **Query Processing**: User questions are converted to embeddings
4. **Retrieval**: Similar document chunks are retrieved using vector similarity search
5. **Generation**: Retrieved context is passed to the LLM to generate accurate answers
6. **Response**: The answer is displayed to the user

## 🎯 System Constraints

The RAG assistant operates with these guidelines:
- Only answers questions based on loaded documents
- Responds in under 80 words for concise answers
- Explicitly states when information is not available
- Prevents hallucinations by grounding responses in retrieved context

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Troubleshooting

**Issue**: "No valid API key found"
- **Solution**: Ensure you've created a `.env` file with at least one API key

**Issue**: "No documents found"
- **Solution**: Add `.txt` or `.md` files to the `data/` directory

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 🔮 Future Enhancements

- Web interface using Streamlit or Gradio
- Support for PDF and DOCX files
- Advanced chunking strategies
- Conversation history and memory
- Multi-language support
- Evaluation metrics for answer quality

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using LangChain and ChromaDB**
