"""
Audio Effect Chain
==================
실제 오디오에 이펙트를 적용하는 처리 체인

pedalboard 라이브러리 사용 (Spotify에서 만든 오디오 플러그인 라이브러리)
- 고품질 VST 수준의 이펙트
- Python에서 쉽게 사용 가능
- 실시간 처리도 가능
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import soundfile as sf

# pedalboard - 오디오 이펙트 라이브러리
from pedalboard import (
    Pedalboard,
    Compressor,
    Gain,
    LowShelfFilter,
    HighShelfFilter,
    PeakFilter,
    Delay,
    Reverb,
    Distortion,
    Limiter,
    HighpassFilter,
    LowpassFilter
)
from pedalboard.io import AudioFile


class EffectChain:
    """오디오 이펙트 처리 체인"""
    
    AVAILABLE_EFFECTS = [
        "eq_lowshelf",
        "eq_highshelf", 
        "eq_peak1",
        "eq_peak2",
        "compressor",
        "distortion",
        "delay",
        "reverb",
        "limiter"
    ]
    
    def __init__(self):
        """이펙트 체인 초기화"""
        pass
    
    def get_available_effects(self) -> List[str]:
        """사용 가능한 이펙트 목록 반환"""
        return self.AVAILABLE_EFFECTS.copy()
    
    def process(
        self, 
        input_path: str, 
        output_path: str, 
        parameters: Dict[str, float]
    ) -> None:
        """
        오디오 파일에 이펙트 체인 적용
        
        Args:
            input_path: 입력 오디오 파일 경로
            output_path: 출력 오디오 파일 경로
            parameters: 이펙터 파라미터 딕셔너리
        """
        # 오디오 파일 읽기
        audio, sample_rate = sf.read(input_path)
        
        # 모노면 스테레오로 변환 (일부 이펙트가 스테레오 필요)
        if len(audio.shape) == 1:
            audio = np.column_stack([audio, audio])
        
        # float32로 변환
        audio = audio.astype(np.float32)
        
        # 이펙트 체인 구성
        board = self._build_pedalboard(parameters, sample_rate)
        
        # 이펙트 적용
        processed = board(audio, sample_rate)
        
        # Wet/Dry 믹스 적용
        wet_mix = parameters.get("final_wet_mix", 0.5)
        final_audio = (1 - wet_mix) * audio + wet_mix * processed
        
        # 클리핑 방지
        final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # 출력 파일 저장
        sf.write(output_path, final_audio, sample_rate)
        
        print(f"[EffectChain] 처리 완료: {output_path}")
    
    def _build_pedalboard(
        self, 
        params: Dict[str, float], 
        sample_rate: int
    ) -> Pedalboard:
        """
        파라미터로부터 pedalboard 이펙트 체인 구성
        """
        effects = []
        
        # === EQ Section ===
        
        # Low Shelf EQ
        if params.get("eq_lowshelf_gain", 0) != 0:
            effects.append(
                LowShelfFilter(
                    cutoff_frequency_hz=params.get("eq_lowshelf_freq", 200),
                    gain_db=params.get("eq_lowshelf_gain", 0),
                    q=0.707
                )
            )
        
        # High Shelf EQ
        if params.get("eq_highshelf_gain", 0) != 0:
            effects.append(
                HighShelfFilter(
                    cutoff_frequency_hz=params.get("eq_highshelf_freq", 8000),
                    gain_db=params.get("eq_highshelf_gain", 0),
                    q=0.707
                )
            )
        
        # Peak EQ 1
        if params.get("eq_peak1_gain", 0) != 0:
            effects.append(
                PeakFilter(
                    cutoff_frequency_hz=params.get("eq_peak1_freq", 1000),
                    gain_db=params.get("eq_peak1_gain", 0),
                    q=params.get("eq_peak1_q", 1.0)
                )
            )
        
        # Peak EQ 2
        if params.get("eq_peak2_gain", 0) != 0:
            effects.append(
                PeakFilter(
                    cutoff_frequency_hz=params.get("eq_peak2_freq", 3000),
                    gain_db=params.get("eq_peak2_gain", 0),
                    q=params.get("eq_peak2_q", 1.0)
                )
            )
        
        # === Dynamics Section ===
        
        # Compressor
        threshold = params.get("compressor_threshold", -24)
        ratio = params.get("compressor_ratio", 4.0)
        if ratio > 1.0:
            effects.append(
                Compressor(
                    threshold_db=threshold,
                    ratio=ratio,
                    attack_ms=params.get("compressor_attack", 5),
                    release_ms=params.get("compressor_release", 50)
                )
            )
            
            # Makeup Gain
            makeup = params.get("compressor_makeup", 0)
            if makeup != 0:
                effects.append(Gain(gain_db=makeup))
        
        # === Distortion Section ===
        
        distortion_amount = params.get("distortion_amount", 0)
        if distortion_amount > 0:
            # pedalboard의 Distortion은 0-100 범위
            effects.append(
                Distortion(drive_db=distortion_amount * 40)  # 0-1 -> 0-40dB
            )
            
            # Distortion 후 톤 조절 (Tone = LPF)
            tone = params.get("distortion_tone", 0.5)
            lpf_freq = 2000 + tone * 10000  # 2kHz ~ 12kHz
            effects.append(
                LowpassFilter(cutoff_frequency_hz=lpf_freq)
            )
        
        # === Time-based Effects Section ===
        
        # Delay
        delay_mix = params.get("delay_mix", 0)
        if delay_mix > 0:
            delay_time_ms = params.get("delay_time", 250)
            effects.append(
                Delay(
                    delay_seconds=delay_time_ms / 1000,
                    feedback=params.get("delay_feedback", 0.3),
                    mix=delay_mix
                )
            )
        
        # Reverb
        reverb_wet = params.get("reverb_wet_dry", 0)
        if reverb_wet > 0:
            effects.append(
                Reverb(
                    room_size=params.get("reverb_room_size", 0.5),
                    damping=params.get("reverb_damping", 0.5),
                    wet_level=reverb_wet,
                    dry_level=1 - reverb_wet,
                    width=1.0
                )
            )
        
        # === Output Section ===
        
        # Limiter (클리핑 방지)
        effects.append(
            Limiter(
                threshold_db=-1.0,
                release_ms=100
            )
        )
        
        return Pedalboard(effects)
    
    def process_realtime(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """
        실시간 오디오 청크 처리 (스트리밍용)
        
        Args:
            audio_chunk: 오디오 데이터 배열
            sample_rate: 샘플레이트
            parameters: 이펙터 파라미터
            
        Returns:
            처리된 오디오 청크
        """
        if len(audio_chunk.shape) == 1:
            audio_chunk = np.column_stack([audio_chunk, audio_chunk])
        
        audio_chunk = audio_chunk.astype(np.float32)
        
        board = self._build_pedalboard(parameters, sample_rate)
        processed = board(audio_chunk, sample_rate)
        
        wet_mix = parameters.get("final_wet_mix", 0.5)
        final = (1 - wet_mix) * audio_chunk + wet_mix * processed
        
        return np.clip(final, -1.0, 1.0)
