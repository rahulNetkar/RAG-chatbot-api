from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import chromadb

# from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader


def create_doc(file):
    try:
        loader = PyPDFLoader(file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        docs = text_splitter.split_documents(documents=documents)

        return docs
    except Exception as e:
        return e


def generate_index(file):
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        docs = create_doc(file=file)

        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory="db"
        )
        vectordb.persist()
        return

    except Exception as e:
        return e


def get_index():
    persist_directory = "db"

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    # response = vectordb.similarity_search(f"{query}")

    return vectordb


def find_match(input):
    index = get_index()

    response = index.similarity_search(f"{input}")

    return response[0].page_content


def construct_prompt(query: str, context: str):
    information = "\n\n".join(context)

    system_prompt = """

    You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report.
    You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information.
    Otherwise say 'I do not know'.
    
    """

    user_prompt = f"Question: {query}. \n Information: {information}. \n Answer:"

    prompt = f"{system_prompt} \n {user_prompt}"

    return prompt
