# MagicPath AI Vocal Effects Server

AI 기반 보컬 이펙트 처리 서버

## 🎯 기능

- **POST /process**: 오디오 파일 + 텍스트 명령 → 처리된 오디오 파일 반환
- **POST /predict**: 오디오 파일 + 텍스트 명령 → 이펙터 파라미터 JSON 반환
- **GET /health**: 서버 상태 확인

## 📁 프로젝트 구조

```
magicpath-server/
├── main.py                 # FastAPI 메인 서버
├── models/
│   ├── __init__.py
│   └── ai_effector.py      # AI 모델 래퍼 (CLAP + LLM)
├── audio_processing/
│   ├── __init__.py
│   └── effect_chain.py     # 실제 오디오 이펙트 처리
├── checkpoints/            # AI 모델 체크포인트 (추가 예정)
├── requirements.txt        # Python 의존성
├── Dockerfile             # 컨테이너 빌드
├── railway.toml           # Railway 배포 설정
└── render.yaml            # Render 배포 설정
```

## 🚀 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 서버 실행
uvicorn main:app --reload --port 8000

# 3. API 문서 확인
# http://localhost:8000/docs
```

## 🌐 API 사용 예시

### 오디오 처리 요청

```bash
curl -X POST "http://localhost:8000/process" \
  -F "audio=@my_vocal.wav" \
  -F "prompt=make it warm" \
  --output processed.wav
```

### 파라미터만 예측

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@my_vocal.wav" \
  -F "prompt=bright modern"
```

응답:
```json
{
  "status": "success",
  "prompt": "bright modern",
  "parameters": {
    "eq_lowshelf_gain": -2.0,
    "eq_highshelf_gain": 4.0,
    "compressor_ratio": 6.0,
    ...
  }
}
```

## ☁️ 배포

### Railway

1. [Railway](https://railway.app) 가입
2. "New Project" → "Deploy from GitHub repo"
3. 이 레포지토리 연결
4. 자동 배포 완료

### Render

1. [Render](https://render.com) 가입
2. "New Web Service" → GitHub 연결
3. 이 레포지토리 선택
4. "Docker" 환경 선택
5. 배포

## 🔧 AI 모델 추가 (추후)

1. `checkpoints/` 폴더에 모델 파일 추가:
   - `lora_weights/` - LoRA 가중치
   - `clap_model/` - CLAP 인코더

2. `requirements.txt`에서 AI 의존성 주석 해제:
   ```
   torch==2.1.2
   transformers==4.36.2
   peft==0.7.1
   laion-clap==1.1.4
   ```

3. `models/ai_effector.py`에서 실제 모델 로딩 코드 활성화

## 📋 지원 이펙트

| 이펙트 | 파라미터 |
|--------|----------|
| EQ Low Shelf | gain, freq |
| EQ High Shelf | gain, freq |
| EQ Peak (x2) | gain, freq, q |
| Compressor | threshold, ratio, attack, release, makeup |
| Distortion | amount, tone |
| Delay | time, feedback, mix |
| Reverb | room_size, damping, wet_dry |
| Limiter | (자동 적용) |

## 🎨 프리셋

AI 모델이 없을 때 사용 가능한 프리셋:
- `warm` - 따뜻한 빈티지 사운드
- `bright` - 밝고 현대적인 사운드
- `radio` - 라디오/전화 느낌
- `spacey` - 공간감 있는 리버브
- `aggressive` - 공격적인 사운드
- `clean` - 깨끗한 자연스러운 사운드
