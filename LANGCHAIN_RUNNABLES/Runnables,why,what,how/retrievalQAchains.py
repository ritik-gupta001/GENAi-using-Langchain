from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitte
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load the document
loader = TextLoader("docs.txt") # Ensure docs.txt exists
documents = loader. load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in FAISS (Vector DB)
vectorstore = FAISS.from_documents (docs, OpenAIEmbeddings())

# Create a retriever (this fetches relevant documents) 
retriever = vectorstore.as_retriever()

# Initialize LLM 
l1m = OpenAI(mode1_name="gpt-3.5-turbo", temperature=0.7)

# Create RetrievalQAChain
qa_chain = RetrievalQA.from_chain_type(l1m=l1m, retriever=retriever)

# Ask a question
query = "What are the key takeaways from the document?"
answer = qa_chain.run(query)

print("Answer:", answer)