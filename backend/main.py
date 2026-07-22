import json
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Gesto Engine Backend API",
    description="手势识别系统后端服务：支持手势样本存储与聚合字典格式同步",
    version="1.1.0"
)

# 配置跨域资源共享 (CORS)，允许前端网页跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "samples.json"

# ==================== 1. Pydantic 数据校验模型 ====================

class SamplePayload(BaseModel):
    # 接收前端传过来的聚合字典结构：{ "g_xxx": { name, img, samples }, "neutral": { ... } }
    samples: Dict[str, Any] = Field(..., description="以手势标签为 Key 的样本聚合字典")

# ==================== 2. RESTful API 接口定义 ====================

@app.get("/api/samples", summary="获取全量手势样本")
def get_samples():
    """
    读取并返回 samples.json 中的所有样本数据（返回聚合字典格式）
    """
    try:
        if not os.path.exists(DATA_FILE):
            return {}
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"后端读取 samples.json 发生异常: {str(e)}"
        )

@app.post("/api/samples", summary="全量更新/同步手势样本库")
def save_samples(payload: SamplePayload):
    """
    接收前端提交的手势聚合字典，直接写入 samples.json
    """
    try:
        samples_dict = payload.samples
        
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(samples_dict, f, ensure_ascii=False, indent=2)
            
        return {
            "status": "success", 
            "message": f"成功保存手势样本至后端",
            "count": len(samples_dict)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"后端写入 samples.json 发生异常: {str(e)}"
        )