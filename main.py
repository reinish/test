"""
MagicPath AI Vocal Effects Server
=================================
Dry 보컬 파일을 받아서 AI가 이펙터 파라미터를 예측하고,
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

app = FastAPI(
    title="MagicPath AI Vocal Effects",
    description="AI-powered vocal effect processing server",
    version="1.0.0"
)

# CORS 설정 - React 앱에서 접근 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체 초기화
ai_effector = AIEffector()
effect_chain = EffectChain()

# 임시 파일 저장 경로
TEMP_DIR = Path(tempfile.gettempdir()) / "magicpath"
TEMP_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "message": "MagicPath AI Vocal Effects Server",
        "endpoints": {
            "POST /process": "오디오 파일 처리",
            "POST /predict": "파라미터만 예측 (오디오 처리 없음)",
            "GET /health": "서버 상태 확인"
        }
    }


@app.get("/health")
async def health_check():
    """서버 및 모델 상태 확인"""
    return {
        "status": "healthy",
        "ai_model_loaded": ai_effector.is_loaded(),
        "supported_effects": effect_chain.get_available_effects()
    }


@app.post("/predict")
async def predict_parameters(
    audio: UploadFile = File(..., description="Dry 보컬 오디오 파일"),
    prompt: str = Form("", description="텍스트 명령 (예: 'make it warm')")
):
    """
    AI 모델로 이펙터 파라미터만 예측 (오디오 처리 없이)
    
    - audio: wav, mp3 등 오디오 파일
    - prompt: 원하는 사운드 설명 (예: "warm vintage", "bright modern")
    
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
            "parameters": parameters
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_audio(
    audio: UploadFile = File(..., description="Dry 보컬 오디오 파일"),
    prompt: str = Form("", description="텍스트 명령 (예: 'make it warm')")
):
    """
    AI가 예측한 파라미터로 실제 오디오 처리
    
    - audio: wav, mp3 등 오디오 파일
    - prompt: 원하는 사운드 설명 (예: "warm vintage", "bright modern")
    
    Returns: 처리된 오디오 파일 (wav)
    """
    try:
        # 임시 파일 경로 생성
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input_{audio.filename}"
        output_path = TEMP_DIR / f"{file_id}_output.wav"
        
        # 입력 파일 저장
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Step 1: AI 모델로 파라미터 예측
        parameters = ai_effector.predict(
            audio_path=str(input_path),
            text_prompt=prompt
        )
        
        # Step 2: 이펙터 체인으로 오디오 처리
        effect_chain.process(
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=parameters
        )
        
        # 입력 파일 삭제 (출력 파일은 응답 후 삭제)
        os.remove(input_path)
        
        # 처리된 오디오 반환
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"processed_{audio.filename.rsplit('.', 1)[0]}.wav",
            background=None  # 파일 전송 후 삭제하려면 BackgroundTask 사용
        )
        
    except Exception as e:
        # 에러 시 임시 파일 정리
        if input_path.exists():
            os.remove(input_path)
        if output_path.exists():
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
