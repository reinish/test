"""
Audio Encoder for MagicPath Server
===================================
CLAP 모델을 사용하여 오디오 파일에서 특징 벡터 추출
DiffVox LLM과 동일한 인코더 사용
"""

import torch
import numpy as np
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")


class AudioEncoder:
    """CLAP 기반 오디오 인코더"""
    
    def __init__(
        self, 
        output_dim: int = 64, 
        reduction_method: str = "pool",
        model_name: str = "laion/larger_clap_general"
    ):
        """
        오디오 인코더 초기화
        
        Args:
            output_dim: 출력 특징 차원 (기본 64)
            reduction_method: 차원 축소 방법 ("pool", "pca", "linear")
            model_name: CLAP 모델 이름
        """
        self.output_dim = output_dim
        self.reduction_method = reduction_method
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.processor = None
        self.projection = None
        
        self._load_model()
    
    def _load_model(self):
        """CLAP 모델 로드"""
        try:
            from transformers import ClapModel, ClapProcessor
            
            print(f"[AudioEncoder] CLAP 모델 로딩 중: {self.model_name}")
            
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # CLAP 출력 차원 확인 (보통 512)
            clap_dim = self.model.config.projection_dim
            print(f"[AudioEncoder] CLAP 출력 차원: {clap_dim}")
            
            # 차원 축소를 위한 projection layer
            if self.reduction_method == "linear" and clap_dim != self.output_dim:
                self.projection = torch.nn.Linear(clap_dim, self.output_dim)
                self.projection = self.projection.to(self.device)
                print(f"[AudioEncoder] Linear projection: {clap_dim} → {self.output_dim}")
            
            print("[AudioEncoder] ✅ 모델 로드 완료")
            
        except ImportError:
            print("[AudioEncoder] ❌ transformers 미설치")
            print("   pip install transformers")
        except Exception as e:
            print(f"[AudioEncoder] ❌ 모델 로드 실패: {e}")
    
    def get_audio_features(self, audio_path: str) -> List[float]:
        """
        오디오 파일에서 특징 벡터 추출
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            특징 벡터 (output_dim 차원)
        """
        if self.model is None:
            print("[AudioEncoder] 모델이 로드되지 않음")
            return []
        
        try:
            import librosa
            
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=48000, mono=True)
            
            # CLAP 입력 준비
            inputs = self.processor(
                audios=audio,
                sampling_rate=48000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 특징 추출
            with torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)
            
            # CPU로 이동
            features = audio_features.squeeze().cpu().numpy()
            
            # 차원 축소
            features = self._reduce_dimension(features)
            
            return features.tolist()
            
        except Exception as e:
            print(f"[AudioEncoder] 특징 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _reduce_dimension(self, features: np.ndarray) -> np.ndarray:
        """특징 벡터 차원 축소"""
        current_dim = len(features)
        
        if current_dim == self.output_dim:
            return features
        
        if self.reduction_method == "pool":
            # 평균 풀링으로 차원 축소
            if current_dim > self.output_dim:
                pool_size = current_dim // self.output_dim
                remainder = current_dim % self.output_dim
                
                pooled = []
                idx = 0
                for i in range(self.output_dim):
                    size = pool_size + (1 if i < remainder else 0)
                    pooled.append(np.mean(features[idx:idx+size]))
                    idx += size
                
                return np.array(pooled)
            else:
                # 차원이 작으면 zero-padding
                padded = np.zeros(self.output_dim)
                padded[:current_dim] = features
                return padded
        
        elif self.reduction_method == "linear" and self.projection is not None:
            # Linear projection
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
                projected = self.projection(features_tensor)
                return projected.cpu().numpy()
        
        else:
            # 기본: 앞에서부터 자르기
            return features[:self.output_dim]
    
    def get_text_features(self, text: str) -> List[float]:
        """
        텍스트에서 특징 벡터 추출 (CLAP text encoder)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            특징 벡터
        """
        if self.model is None:
            return []
        
        try:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            features = text_features.squeeze().cpu().numpy()
            features = self._reduce_dimension(features)
            
            return features.tolist()
            
        except Exception as e:
            print(f"[AudioEncoder] 텍스트 특징 추출 실패: {e}")
            return []
