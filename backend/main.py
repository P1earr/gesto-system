import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自动定位目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# 挂载静态文件目录 static/uploads
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

SAMPLES_FILE = os.path.join(BASE_DIR, "samples.json")

class SamplesPayload(BaseModel):
    samples: dict

# 1. 根路由：访问 http://127.0.0.1:8000 直接打开 index.html 网页
@app.get("/")
async def read_index():
    index_path = os.path.join(PARENT_DIR, "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path)

# 2. 图片上传接口
@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只允许上传图片文件")
    
    ext = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    return {"url": f"/static/uploads/{filename}"}

# 3. 样本数据保存与读取接口
@app.get("/api/samples")
async def get_samples():
    if os.path.exists(SAMPLES_FILE):
        with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

@app.post("/api/samples")
async def save_samples(payload: SamplesPayload):
    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload.samples, f, ensure_ascii=False, indent=2)
    return {"status": "ok"}