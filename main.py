"""
MagicPath AI Vocal Effects Server - DiffVox LLM 통합 버전
=========================================================
Dry 보컬 파일을 받아서 학습된 AI가 이펙터 파라미터를 예측하고,
실제로 이펙트를 적용한 오디오를 반환하는 서버
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import uuid
from pathlib import Path

# 내부 모듈
from models.ai_effector import AIEffector
from audio_processing.effect_chain import EffectChain

# ============================================
# 설정
# ============================================

# 학습된 모델 경로 (Hugging Face 레포 또는 로컬 경로)
MODEL_PATH = os.environ.get("DIFFVOX_MODEL_PATH", "heybaeheef/KU_SW_Academy")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen3-8B")
AUDIO_FEATURE_DIM = int(os.environ.get("AUDIO_FEATURE_DIM", "64"))
USE_HUGGINGFACE = os.environ.get("USE_HUGGINGFACE", "true").lower() == "true"

# ============================================
# FastAPI 앱 초기화
# ============================================

app = FastAPI(
    title="MagicPath AI Vocal Effects",
    description="AI-powered vocal effect processing server (DiffVox LLM 통합)",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체 초기화
print("=" * 60)
print("MagicPath AI Vocal Effects Server v2.0")
print("=" * 60)
print(f"Model Path: {MODEL_PATH}")
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"Audio Feature Dim: {AUDIO_FEATURE_DIM}")
print(f"Use Hugging Face: {USE_HUGGINGFACE}")
print("=" * 60)

ai_effector = AIEffector(
    model_path=MODEL_PATH,
    base_model_name=BASE_MODEL_NAME,
    audio_feature_dim=AUDIO_FEATURE_DIM,
    use_huggingface=USE_HUGGINGFACE
)
effect_chain = EffectChain()

# 임시 파일 저장 경로
TEMP_DIR = Path(tempfile.gettempdir()) / "magicpath"
TEMP_DIR.mkdir(exist_ok=True)


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """서버 정보"""
    return {
        "status": "running",
        "message": "MagicPath AI Vocal Effects Server v2.0 (DiffVox LLM)",
        "ai_model_loaded": ai_effector.is_loaded(),
        "endpoints": {
            "POST /process": "오디오 파일 처리 후 반환",
            "POST /predict": "파라미터만 예측 (JSON)",
            "GET /health": "서버 상태 확인"
        }
    }


@app.get("/health")
async def health_check():
    """서버 및 모델 상태 확인"""
    return {
        "status": "healthy",
        "ai_model_loaded": ai_effector.is_loaded(),
        "supported_effects": effect_chain.get_available_effects(),
        "model_path": MODEL_PATH,
        "base_model": BASE_MODEL_NAME
    }


@app.post("/predict")
async def predict_parameters(
    audio: UploadFile = File(..., description="Dry 보컬 오디오 파일"),
    prompt: str = Form("", description="텍스트 명령 (예: 'warm', 'bright')")
):
    """
    AI 모델로 이펙터 파라미터 예측 (오디오 처리 없이)
    
    - audio: wav, mp3 등 오디오 파일
    - prompt: 원하는 사운드 설명
    
    Returns: 예측된 이펙터 파라미터 JSON
    """
    try:
        # 임시 파일로 저장
        input_path = TEMP_DIR / f"{uuid.uuid4()}_{audio.filename}"
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # AI 모델로 파라미터 예측
        parameters = ai_effector.predict(
            audio_path=str(input_path),
            text_prompt=prompt
        )
        
        # 임시 파일 삭제
        os.remove(input_path)
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "ai_model_used": ai_effector.is_loaded(),
            "parameters": parameters
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_audio(
    audio: UploadFile = File(..., description="Dry 보컬 오디오 파일"),
    prompt: str = Form("", description="텍스트 명령 (예: 'warm', 'bright')")
):
    """
    AI가 예측한 파라미터로 실제 오디오 처리
    
    - audio: wav, mp3 등 오디오 파일
    - prompt: 원하는 사운드 설명
    
    Returns: 처리된 오디오 파일 (wav)
    """
    input_path = None
    output_path = None
    
    try:
        # 임시 파일 경로 생성
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input_{audio.filename}"
        output_path = TEMP_DIR / f"{file_id}_output.wav"
        
        # 입력 파일 저장
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        print(f"[Process] 입력 파일: {input_path}")
        print(f"[Process] 프롬프트: {prompt}")
        
        # Step 1: AI 모델로 파라미터 예측
        parameters = ai_effector.predict(
            audio_path=str(input_path),
            text_prompt=prompt
        )
        
        print(f"[Process] 예측된 파라미터: {len(parameters)}개")
        
        # Step 2: 이펙터 체인으로 오디오 처리
        effect_chain.process(
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=parameters
        )
        
        # 입력 파일 삭제
        os.remove(input_path)
        
        # 처리된 오디오 반환
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"processed_{audio.filename.rsplit('.', 1)[0]}.wav",
            background=None
        )
        
    except Exception as e:
        # 에러 시 임시 파일 정리
        if input_path and input_path.exists():
            os.remove(input_path)
        if output_path and output_path.exists():
            os.remove(output_path)
        
        print(f"[Process] ❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_with_params")
async def process_audio_with_params(
    audio: UploadFile = File(..., description="Dry 보컬 오디오 파일"),
    prompt: str = Form("", description="텍스트 명령")
):
    """
    오디오 처리 + 사용된 파라미터도 함께 반환
    
    Returns: JSON (처리된 오디오 URL + 파라미터)
    """
    input_path = None
    output_path = None
    
    try:
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input_{audio.filename}"
        output_path = TEMP_DIR / f"{file_id}_output.wav"
        
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # AI 파라미터 예측
        parameters = ai_effector.predict(
            audio_path=str(input_path),
            text_prompt=prompt
        )
        
        # 오디오 처리
        effect_chain.process(
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=parameters
        )
        
        os.remove(input_path)
        
        # Base64 인코딩으로 오디오 반환 (또는 URL)
        import base64
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(output_path)
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "ai_model_used": ai_effector.is_loaded(),
            "parameters": parameters,
            "audio_base64": audio_base64,
            "audio_format": "wav"
        })
        
    except Exception as e:
        if input_path and input_path.exists():
            os.remove(input_path)
        if output_path and output_path.exists():
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
