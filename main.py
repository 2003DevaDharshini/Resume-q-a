import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from io import BytesIO

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

Settings.llm = Gemini(api_key=gemini_api_key, model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(api_key=gemini_api_key, model="models/embedding-001")

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    documents = SimpleDirectoryReader(input_files=["resume.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=1)

app = FastAPI(
    title="Resume Q&A API",
    description="Ask questions about my resume using LlamaIndex + Gemini Flash",
    version="3.0"
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    default_json = json.dumps({"query": "Tell me about yourself"}, indent=4)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "output": None,
        "input_json": default_json
    })


@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, input_json: str = Form(...)):
    try:
        data = json.loads(input_json)
        query = data.get("query", "")
        if not query:
            raise ValueError("Missing 'query' field in JSON.")
        response = query_engine.query(query)
        output = {"query": query, "response": str(response)}
    except Exception as e:
        output = {"error": str(e)}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "output": json.dumps(output, indent=4),
        "input_json": input_json
    })


@app.post("/download", response_class=StreamingResponse)
async def download_response(request: Request):
    form = await request.form()
    output_data = form.get("output_json", "{}")
    buffer = BytesIO()
    buffer.write(output_data.encode())
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/json", headers={
        "Content-Disposition": "attachment; filename=response.json"
    })
