import json
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="Gesto Engine Backend API",
    description="手势识别系统后端服务：支持手势样本存储与输入格式校验",
    version="1.0.0"
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

# ==================== 1. Pydantic 输入数据校验模型 ====================

class SampleItem(BaseModel):
    label: str = Field(..., min_length=1, description="手势唯一标识 (Label)")
    name: str = Field(..., min_length=1, description="手势显示名称")
    img: Optional[str] = Field(default="", description="绑定的 Emoji 或 Base64 图片数据")
    features: List[float] = Field(..., description="MediaPipe 提取的特征向量坐标组")

    # 逻辑健壮性得分点：校验特征向量维度与非空状态
    @field_validator("features")
    def validate_features(cls, v):
        if not v or len(v) == 0:
            raise ValueError("特征向量 (features) 不能为空")
        return v

class SampleListPayload(BaseModel):
    samples: List[SampleItem]

# ==================== 2. RESTful API 接口定义 ====================

@app.get("/api/samples", response_model=List[SampleItem], summary="获取全量手势样本")
def get_samples():
    """
    读取并返回 samples.json 中的所有样本数据
    """
    try:
        if not os.path.exists(DATA_FILE):
            return []
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"后端读取 samples.json 发生异常: {str(e)}"
        )

@app.post("/api/samples", summary="全量更新/同步手势样本库")
def save_samples(payload: SampleListPayload):
    """
    接收前端提交的手势数组，经过 Pydantic 严格校验后写入 samples.json
    """
    try:
        # 将 Pydantic 对象转为字典列表
        samples_data = [item.model_dump() for item in payload.samples]
        
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(samples_data, f, ensure_ascii=False, indent=2)
            
        return {
            "status": "success", 
            "message": f"成功保存 {len(samples_data)} 条手势样本至后端",
            "count": len(samples_data)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"后端写入 samples.json 发生异常: {str(e)}"
        )