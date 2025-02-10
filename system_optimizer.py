import psutil
import numpy as np
import cv2
from enum import Enum, auto
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque
import yaml
from pathlib import Path
import GPUtil

class ProcessingMode(Enum):
    LIGHT = auto()     # Solo detección básica (personas)
    BALANCED = auto()  # Detección + análisis postural
    FULL = auto()      # Todas las features

class SystemOptimizer:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el optimizador del sistema.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Inicializar métricas
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.current_mode = ProcessingMode.BALANCED
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config['optimization']['max_threads']
        )
        
        # Métricas del sistema
        self.metrics = {
            'fps': 0.0,
            'gpu_temp': 0.0,
            'gpu_usage': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }
        
        # ROI settings
        self.roi_margin = self.config['optimization']['roi_margin']
        
    def _load_config(self, config_path: str) -> dict:
        """Carga la configuración desde YAML."""
        if not Path(config_path).exists():
            # Crear configuración por defecto
            config = {
                'optimization': {
                    'resolution': '1280x720',
                    'fp16_enabled': True,
                    'max_threads': 4,
                    'roi_margin': 30,
                    'min_fps': 25
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            return config
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def update_metrics(self) -> None:
        """Actualiza las métricas del sistema."""
        # Calcular FPS
        current_time = time.time()
        self.fps_buffer.append(1.0 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        self.metrics['fps'] = np.mean(self.fps_buffer)
        
        # Métricas de GPU si está disponible
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Usar primera GPU
            self.metrics['gpu_temp'] = gpu.temperature
            self.metrics['gpu_usage'] = gpu.load * 100
        
        # Métricas de CPU y memoria
        self.metrics['cpu_usage'] = psutil.cpu_percent()
        self.metrics['memory_usage'] = psutil.virtual_memory().percent
        
        # Ajustar modo según métricas
        self._adjust_processing_mode()
    
    def _adjust_processing_mode(self) -> None:
        """Ajusta el modo de procesamiento según las métricas."""
        if self.metrics['fps'] < self.config['optimization']['min_fps']:
            self.current_mode = ProcessingMode.LIGHT
        elif self.metrics['gpu_temp'] > 80:
            self.current_mode = ProcessingMode.BALANCED
        elif (self.metrics['gpu_usage'] < 70 and 
              self.metrics['cpu_usage'] < 70 and 
              self.metrics['fps'] > 30):
            self.current_mode = ProcessingMode.FULL
    
    def process_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Procesa una región de interés del frame.
        
        Args:
            frame: Frame completo
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            ROI recortada
        """
        x1, y1, x2, y2 = bbox
        margin = self.roi_margin
        
        # Ajustar márgenes a límites del frame
        y1_roi = max(0, y1 - margin)
        y2_roi = min(frame.shape[0], y2 + margin)
        x1_roi = max(0, x1 - margin)
        x2_roi = min(frame.shape[1], x2 + margin)
        
        return frame[y1_roi:y2_roi, x1_roi:x2_roi]
    
    def draw_metrics(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja las métricas en el frame.
        
        Args:
            frame: Frame a dibujar
            
        Returns:
            Frame con métricas
        """
        # Crear panel semi-transparente
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (250, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Dibujar métricas
        metrics_text = [
            f"FPS: {self.metrics['fps']:.1f}",
            f"Mode: {self.current_mode.name}",
            f"GPU: {self.metrics['gpu_usage']:.1f}% ({self.metrics['gpu_temp']:.1f}°C)",
            f"CPU: {self.metrics['cpu_usage']:.1f}%",
            f"RAM: {self.metrics['memory_usage']:.1f}%"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (10, 25 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
        
        # Dibujar alertas si es necesario
        if self.metrics['fps'] < self.config['optimization']['min_fps']:
            cv2.putText(frame, "LOW FPS!", (10, panel_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 255), 2)
        
        if self.metrics['gpu_temp'] > 80:
            cv2.putText(frame, "HIGH GPU TEMP!", (120, panel_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 255), 2)
        
        return frame
    
    def should_process_feature(self, feature: str) -> bool:
        """
        Determina si una característica debe procesarse según el modo actual.
        
        Args:
            feature: Nombre de la característica
            
        Returns:
            True si debe procesarse
        """
        feature_modes = {
            'pose': [ProcessingMode.BALANCED, ProcessingMode.FULL],
            'emotion': [ProcessingMode.FULL],
            'objects': [ProcessingMode.FULL],
            'person': [ProcessingMode.LIGHT, ProcessingMode.BALANCED, ProcessingMode.FULL]
        }
        
        return self.current_mode in feature_modes.get(feature, []) 