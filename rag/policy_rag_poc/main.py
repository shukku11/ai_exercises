import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# 1. Setup Azure OpenAI values (use your Foundry resource values)
os.environ.setdefault("AZURE_OPENAI_API_KEY", "")  # Add your API key here or set it in your environment variables
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://base-model11.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# 2. Load and Chunk the PDF
pdf_path = os.path.join(os.path.dirname(__file__), "docs", "FY26_US_Incentive_Guide.pdf")
if not os.path.isfile(pdf_path):
    raise FileNotFoundError("Add your PDF at rag/policy_rag_poc/docs/FY26_US_Incentive_Guide.pdf and run again.")

loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split the text so it fits in the model's memory (context window)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# 3. Create Embeddings and Store in Vector DB
# Use the exact embedding deployment name from Foundry
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 4. Setup the chat model using your Foundry deployment name
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0,
)

# 5. The RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_db.as_retriever()
)

# 6. Ask Questions in a loop (type 'exit' to stop)
print("RAG assistant ready. Type your question, or 'exit' to quit.")

while True:
    query = input("Enter your question: ").strip()

    if query.lower() in {"exit", "quit"}:
        print("Exiting.")
        break

    if not query:
        print("Please enter a question.")
        continue

    response = rag_chain.invoke(query)
    print(response["result"])
    print("-" * 60)
