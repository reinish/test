# MagicPath AI Vocal Effects Server v2.0

**DiffVox LLM 통합 버전** - 학습된 AI가 보컬 특성을 분석하여 최적의 이펙터 파라미터를 생성합니다.

## 🎯 기능

- **POST /process**: 오디오 파일 + 텍스트 → AI가 파라미터 예측 → 처리된 오디오 반환
- **POST /predict**: 오디오 파일 + 텍스트 → 이펙터 파라미터 JSON 반환
- **POST /process_with_params**: 처리된 오디오 + 파라미터 JSON 함께 반환
- **GET /health**: 서버 및 AI 모델 상태 확인

## 📁 프로젝트 구조

```
magicpath-server/
├── main.py                      # FastAPI 메인 서버
├── models/
│   ├── __init__.py
│   ├── ai_effector.py           # DiffVox LLM 래퍼
│   └── audio_encoder.py         # CLAP 오디오 인코더
├── audio_processing/
│   ├── __init__.py
│   └── effect_chain.py          # pedalboard 이펙트 체인
├── checkpoints/                 # AI 모델 체크포인트
│   └── diffvox_model/           # 학습된 LoRA 모델 (추가 필요)
├── requirements.txt
└── README.md
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. AI 모델 추가 (선택사항)

학습된 DiffVox LoRA 모델을 `checkpoints/diffvox_model/` 폴더에 복사:

```
checkpoints/
└── diffvox_model/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

### 3. 서버 실행

```bash
# 기본 실행
uvicorn main:app --reload --port 8000

# 환경변수로 모델 경로 지정
DIFFVOX_MODEL_PATH=./checkpoints/diffvox_model uvicorn main:app --port 8000
```

### 4. API 문서 확인

http://localhost:8000/docs

## 🌐 API 사용 예시

### 파라미터 예측

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@my_vocal.wav" \
  -F "prompt=warm"
```

응답:
```json
{
  "status": "success",
  "prompt": "warm",
  "ai_model_used": true,
  "parameters": {
    "eq_lowshelf_gain": 5.2,
    "eq_highshelf_gain": -1.3,
    "delay_mix": 0.15,
    ...
  }
}
```

### 오디오 처리

```bash
curl -X POST "http://localhost:8000/process" \
  -F "audio=@my_vocal.wav" \
  -F "prompt=bright modern" \
  --output processed.wav
```

### 오디오 + 파라미터 함께 받기

```bash
curl -X POST "http://localhost:8000/process_with_params" \
  -F "audio=@my_vocal.wav" \
  -F "prompt=warm"
```

응답:
```json
{
  "status": "success",
  "parameters": {...},
  "audio_base64": "UklGRv4...",
  "audio_format": "wav"
}
```

## 🔧 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DIFFVOX_MODEL_PATH` | `./checkpoints/diffvox_model` | LoRA 모델 경로 |
| `BASE_MODEL_NAME` | `Qwen/Qwen3-8B` | 베이스 LLM 모델 |
| `AUDIO_FEATURE_DIM` | `64` | CLAP 출력 차원 |

## 📊 파라미터 매핑

DiffVox LLM 출력 → MagicPath 웹 형식으로 자동 변환됩니다:

| DiffVox LLM | MagicPath 웹 |
|-------------|--------------|
| `eq_lowshelf.params.gain` | `eq_lowshelf_gain` |
| `eq_peak1.params.parametrizations.freq.original` | `eq_peak1_freq` |
| `delay.mix` | `delay_mix` |
| ... | ... |

## ⚠️ AI 모델 없이 실행

AI 모델이 없으면 **프리셋 모드**로 동작합니다:
- `warm`, `bright`, `radio`, `spacey`, `aggressive`, `clean` 키워드 지원
- 키워드가 매칭되지 않으면 기본값 반환

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

## 🔗 프론트엔드 연동

React 앱에서 사용 예시:

```javascript
const processAudio = async (audioFile, prompt) => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('prompt', prompt);
  
  const response = await fetch('http://localhost:8000/process_with_params', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  
  // 파라미터로 UI 업데이트
  setParameters(data.parameters);
  
  // Base64 오디오 재생
  const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
  const audioUrl = URL.createObjectURL(audioBlob);
  audioPlayer.src = audioUrl;
};
```
