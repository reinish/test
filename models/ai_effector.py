"""
AI Effector Model - DiffVox LLM 통합 버전
==========================================
CLAP 인코더 + 학습된 LLM을 사용하여 오디오에서 이펙터 파라미터를 예측

DiffVox LLM 파라미터 → MagicPath 웹 파라미터 자동 변환
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# AI 모델 관련 import (설치 필요)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[AIEffector] transformers/peft 미설치 - 프리셋 모드로 동작")

# CLAP 인코더 (별도 파일)
try:
    from models.audio_encoder import AudioEncoder
    AUDIO_ENCODER_AVAILABLE = True
except ImportError:
    AUDIO_ENCODER_AVAILABLE = False
    print("[AIEffector] AudioEncoder 미설치 - 프리셋 모드로 동작")


class ParameterMapper:
    """DiffVox LLM 파라미터 ↔ MagicPath 웹 파라미터 변환"""
    
    # DiffVox LLM → MagicPath 웹 매핑
    DIFFVOX_TO_WEB = {
        # EQ Low Shelf
        "eq_lowshelf.params.gain": "eq_lowshelf_gain",
        "eq_lowshelf.params.parametrizations.freq.original": "eq_lowshelf_freq",
        # EQ High Shelf
        "eq_highshelf.params.gain": "eq_highshelf_gain",
        "eq_highshelf.params.parametrizations.freq.original": "eq_highshelf_freq",
        # EQ Peak 1
        "eq_peak1.params.gain": "eq_peak1_gain",
        "eq_peak1.params.parametrizations.freq.original": "eq_peak1_freq",
        "eq_peak1.params.parametrizations.Q.original": "eq_peak1_q",
        # EQ Peak 2
        "eq_peak2.params.gain": "eq_peak2_gain",
        "eq_peak2.params.parametrizations.freq.original": "eq_peak2_freq",
        "eq_peak2.params.parametrizations.Q.original": "eq_peak2_q",
        # Delay
        "delay.delay_time": "delay_time",
        "delay.feedback": "delay_feedback",
        "delay.mix": "delay_mix",
        # Distortion
        "distortion_amount": "distortion_amount",
        # Master
        "final_wet_mix": "final_wet_mix",
    }
    
    # 역방향 매핑
    WEB_TO_DIFFVOX = {v: k for k, v in DIFFVOX_TO_WEB.items()}
    
    # 값 변환 규칙 (정규화된 값 → 실제 값)
    VALUE_TRANSFORMS = {
        # EQ gain: -1~1 → -12~12 dB
        "eq_lowshelf_gain": lambda x: x * 12,
        "eq_highshelf_gain": lambda x: x * 12,
        "eq_peak1_gain": lambda x: x * 12,
        "eq_peak2_gain": lambda x: x * 12,
        # EQ freq: 정규화된 값 → Hz (로그 스케일 역변환 필요할 수 있음)
        "eq_lowshelf_freq": lambda x: 20 * (20000/20) ** ((x + 1) / 2),  # -1~1 → 20~20000
        "eq_highshelf_freq": lambda x: 20 * (20000/20) ** ((x + 1) / 2),
        "eq_peak1_freq": lambda x: 20 * (20000/20) ** ((x + 1) / 2),
        "eq_peak2_freq": lambda x: 20 * (20000/20) ** ((x + 1) / 2),
        # Q: -1~1 → 0.1~10
        "eq_peak1_q": lambda x: 0.1 * (10/0.1) ** ((x + 1) / 2),
        "eq_peak2_q": lambda x: 0.1 * (10/0.1) ** ((x + 1) / 2),
        # Delay time: -1~1 → 0~1000 ms
        "delay_time": lambda x: (x + 1) / 2 * 1000,
        # Delay feedback: -1~1 → 0~1
        "delay_feedback": lambda x: (x + 1) / 2,
        # Delay mix: -1~1 → 0~1
        "delay_mix": lambda x: (x + 1) / 2,
        # Distortion: -1~1 → 0~1
        "distortion_amount": lambda x: (x + 1) / 2,
        # Wet mix: -1~1 → 0~1
        "final_wet_mix": lambda x: (x + 1) / 2,
    }
    
    @classmethod
    def diffvox_to_web(cls, diffvox_params: Dict[str, float]) -> Dict[str, float]:
        """DiffVox LLM 출력 → MagicPath 웹 파라미터"""
        web_params = {}
        
        for diffvox_key, value in diffvox_params.items():
            # 키 변환
            if diffvox_key in cls.DIFFVOX_TO_WEB:
                web_key = cls.DIFFVOX_TO_WEB[diffvox_key]
            else:
                # 매핑에 없으면 스킵
                continue
            
            # 값 변환
            if web_key in cls.VALUE_TRANSFORMS:
                try:
                    web_params[web_key] = cls.VALUE_TRANSFORMS[web_key](value)
                except:
                    web_params[web_key] = value
            else:
                web_params[web_key] = value
        
        return web_params


class ParameterParser:
    """LLM 출력에서 파라미터 JSON 추출"""
    
    @staticmethod
    def parse(llm_output: str) -> Optional[Dict]:
        """LLM 출력에서 파라미터 딕셔너리 추출"""
        
        # 방법 1: JSON 블록 찾기
        json_patterns = [
            r'\{[^{}]*\}',
            r'\{(?:[^{}]|\{[^{}]*\})*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, llm_output, re.DOTALL)
            for match in matches:
                try:
                    params = json.loads(match)
                    if isinstance(params, dict) and len(params) > 0:
                        return params
                except json.JSONDecodeError:
                    continue
        
        # 방법 2: key: value 패턴 파싱
        param_pattern = r'"([^"]+)":\s*([-\d.]+)'
        matches = re.findall(param_pattern, llm_output)
        if matches:
            params = {}
            for key, value in matches:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            if params:
                return params
        
        return None


class AIEffector:
    """AI 기반 이펙터 파라미터 예측 모델 - DiffVox LLM 통합"""
    
    # 기본 파라미터
    DEFAULT_PARAMS = {
        "eq_lowshelf_gain": 0.0,
        "eq_lowshelf_freq": 200,
        "eq_highshelf_gain": 0.0,
        "eq_highshelf_freq": 8000,
        "eq_peak1_gain": 0.0,
        "eq_peak1_freq": 1000,
        "eq_peak1_q": 1.0,
        "eq_peak2_gain": 0.0,
        "eq_peak2_freq": 3000,
        "eq_peak2_q": 1.0,
        "compressor_threshold": -24,
        "compressor_ratio": 4.0,
        "compressor_attack": 5,
        "compressor_release": 50,
        "compressor_makeup": 0.0,
        "distortion_amount": 0.0,
        "distortion_tone": 0.5,
        "delay_time": 250,
        "delay_feedback": 0.3,
        "delay_mix": 0.0,
        "reverb_room_size": 0.5,
        "reverb_damping": 0.5,
        "reverb_wet_dry": 0.0,
        "final_wet_mix": 0.5
    }
    
    # 프리셋 (fallback용)
    PRESETS = {
        "warm": {
            "eq_lowshelf_gain": 5.5,
            "eq_lowshelf_freq": 200,
            "eq_highshelf_gain": -1.5,
            "eq_highshelf_freq": 8000,
            "eq_peak1_gain": 2.0,
            "eq_peak1_freq": 400,
            "eq_peak1_q": 1.0,
            "compressor_threshold": -18,
            "compressor_ratio": 3.0,
            "distortion_amount": 0.05,
            "reverb_room_size": 0.4,
            "reverb_wet_dry": 0.15,
            "final_wet_mix": 0.5
        },
        "bright": {
            "eq_lowshelf_gain": -2.0,
            "eq_lowshelf_freq": 150,
            "eq_highshelf_gain": 4.0,
            "eq_highshelf_freq": 6000,
            "eq_peak1_gain": 1.0,
            "eq_peak1_freq": 3000,
            "compressor_threshold": -20,
            "compressor_ratio": 6.0,
            "reverb_room_size": 0.3,
            "reverb_wet_dry": 0.1,
            "final_wet_mix": 0.5
        },
    }
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        base_model_name: str = "Qwen/Qwen3-8B",
        audio_feature_dim: int = 64,
        use_huggingface: bool = True
    ):
        """
        AI 모델 초기화
        
        Args:
            model_path: 학습된 LoRA 모델 경로 (로컬 또는 Hugging Face 레포)
            base_model_name: 베이스 LLM 모델 이름
            audio_feature_dim: 오디오 특징 차원 (CLAP 출력)
            use_huggingface: True면 model_path를 Hugging Face 레포로 간주
        """
        self.model = None
        self.tokenizer = None
        self.audio_encoder = None
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.base_model_name = base_model_name
        self.audio_feature_dim = audio_feature_dim
        self.use_huggingface = use_huggingface
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """학습된 LoRA 모델 로드 (로컬 또는 Hugging Face)"""
        if not TRANSFORMERS_AVAILABLE:
            print("[AIEffector] transformers/peft 미설치")
            return
        
        # 로컬 경로인지 Hugging Face 레포인지 확인
        is_local = os.path.exists(model_path)
        
        if not is_local and not self.use_huggingface:
            print(f"[AIEffector] 로컬 모델 경로 없음: {model_path}")
            return
        
        try:
            if self.use_huggingface and not is_local:
                print(f"[AIEffector] Hugging Face에서 모델 로딩: {model_path}")
            else:
                print(f"[AIEffector] 로컬 모델 로딩: {model_path}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # LoRA 어댑터 적용 (Hugging Face 레포 또는 로컬 경로)
            self.model = PeftModel.from_pretrained(
                base_model,
                model_path,  # Hugging Face 레포 이름 또는 로컬 경로
                is_trainable=False
            )
            self.model.eval()
            
            # 오디오 인코더 로드
            if AUDIO_ENCODER_AVAILABLE:
                self.audio_encoder = AudioEncoder(
                    output_dim=self.audio_feature_dim,
                    reduction_method="pool"
                )
                print("[AIEffector] AudioEncoder 로드 완료")
            
            self.model_loaded = True
            print("[AIEffector] ✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"[AIEffector] ❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def is_loaded(self) -> bool:
        """AI 모델 로드 상태 확인"""
        return self.model_loaded
    
    def predict(self, audio_path: str, text_prompt: str) -> Dict[str, float]:
        """
        오디오와 텍스트로부터 이펙터 파라미터 예측
        
        Args:
            audio_path: 입력 오디오 파일 경로
            text_prompt: 사용자 텍스트 명령
            
        Returns:
            MagicPath 웹 형식의 이펙터 파라미터 딕셔너리
        """
        if self.model_loaded and self.audio_encoder:
            return self._predict_with_model(audio_path, text_prompt)
        else:
            return self._predict_with_preset(text_prompt)
    
    def _predict_with_model(self, audio_path: str, text_prompt: str) -> Dict[str, float]:
        """학습된 DiffVox LLM으로 추론"""
        try:
            # 1. 오디오 특징 추출
            audio_features = self.audio_encoder.get_audio_features(audio_path)
            if not audio_features:
                print("[AIEffector] 오디오 특징 추출 실패, 프리셋 사용")
                return self._predict_with_preset(text_prompt)
            
            # 2. 프롬프트 구성 (train_model.py와 동일한 형식)
            audio_state_str = json.dumps(audio_features)
            prompt = f"""Task: Convert text to audio parameters.
Audio: {audio_state_str}
Text: {text_prompt}
Parameters:"""
            
            # 3. LLM 추론
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            print(f"[AIEffector] LLM 출력: {generated_text[:200]}...")
            
            # 4. 파라미터 파싱
            diffvox_params = ParameterParser.parse(generated_text)
            
            if not diffvox_params:
                print("[AIEffector] 파라미터 파싱 실패, 프리셋 사용")
                return self._predict_with_preset(text_prompt)
            
            # 5. DiffVox → Web 파라미터 변환
            web_params = ParameterMapper.diffvox_to_web(diffvox_params)
            
            # 6. 기본값과 병합
            result = self.DEFAULT_PARAMS.copy()
            result.update(web_params)
            
            print(f"[AIEffector] ✅ AI 파라미터 생성 완료: {len(web_params)}개 파라미터")
            return result
            
        except Exception as e:
            print(f"[AIEffector] 추론 에러: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_with_preset(text_prompt)
    
    def _predict_with_preset(self, text_prompt: str) -> Dict[str, float]:
        """프리셋 기반 파라미터 반환 (fallback)"""
        prompt_lower = text_prompt.lower()
        
        for preset_name, preset_params in self.PRESETS.items():
            if preset_name in prompt_lower:
                print(f"[AIEffector] 프리셋 매칭: '{preset_name}'")
                result = self.DEFAULT_PARAMS.copy()
                result.update(preset_params)
                return result
        
        print("[AIEffector] 프리셋 매칭 실패, 기본값 반환")
        return self.DEFAULT_PARAMS.copy()
