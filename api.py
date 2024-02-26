# uvicorn api:app --reload

# main imports
from fastapi import FastAPI, UploadFile, HTTPException, status
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2 as pdf
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from dotenv import load_dotenv
import os

# Function imports
# ...
from utils import *

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
embedding_function = GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
llm = genai.GenerativeModel("gemini-pro")

# Initialize app
app = FastAPI()

# CORS
origin = [
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


# Health check
@app.get("/healthy")
async def healthy():
    return {"message": "healthy"}


@app.post("/upload-file", status_code=status.HTTP_200_OK)
async def upload_and_generate_index(file: UploadFile):
    if file is not None:
        try:

            loader = pdf.PdfReader(file.file)

            text = ""
            for page in loader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=20
            )
            docs = text_splitter.split_text(text=text)

            tokens = []

            token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=0, tokens_per_chunk=384
            )

            for text in docs:
                tokens += token_splitter.split_text(text=text)

            # initializing chroma http-client
            chroma_client = chromadb.HttpClient(settings=Settings(allow_reset=True))
            chroma_client.reset()

            # creating collection
            collection = chroma_client.create_collection(
                name="document-embeddings",
                embedding_function=embedding_function,
            )

            ids = [str(i) for i in range(len(tokens))]

            collection.add(
                ids=ids,
                documents=tokens,
            )

            return {"message": "Document Index created"}
        except Exception as e:
            raise (e)
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Please upload file",
        )


@app.get("/get-response")
async def get_response(input: str):
    if input is not None:
        try:

            chroma_client = chromadb.HttpClient(settings=Settings(allow_reset=True))

            collection = chroma_client.get_collection(
                name="document-embeddings",
                embedding_function=embedding_function,
            )

            db_response = collection.query(query_texts=[f"{input}"], n_results=3)

            context = db_response["documents"][0]

            prompt = construct_prompt(input, context)

            llm_response = llm.generate_content(prompt)

            return {"message": llm_response.text}
        except Exception as e:
            raise (e)
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Please input your query",
        )
