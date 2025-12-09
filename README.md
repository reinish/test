---
title: DiffVox AI Vocal Effects Server
emoji: 🎤
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# DiffVox AI Vocal Effects Server

AI-powered vocal effect processing server using DiffVox LLM.

## API Endpoints

- `GET /` - Server info
- `GET /health` - Health check
- `POST /predict` - Predict effect parameters
- `POST /process` - Process audio with AI-predicted parameters
- `POST /process_with_params` - Process audio and return parameters + audio

## Usage

```bash
curl -X POST "https://YOUR-SPACE.hf.space/process_with_params" \
  -F "audio=@your_vocal.wav" \
  -F "prompt=warm vintage sound"
```
