from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
import cv2
from dataclasses import dataclass

@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    component_hash: str = ""
    dominant_color: tuple = (0, 0, 0)

class ComponentAnalyzer:
    def __init__(self, model_path: str = 'yolo11n.pt', conf_threshold: float = 0.5):
        """
        Inicializa el analizador de componentes.
        
        Args:
            model_path: Ruta al modelo YOLO11 (opciones: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
            conf_threshold: Umbral de confianza para detecciones
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = {
            0: 'person',
            24: 'backpack',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            32: 'sports ball'
        }

    def _calculate_dominant_color(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """Calcula el color dominante en una región."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        # Redimensionar ROI para procesamiento más rápido
        small_roi = cv2.resize(roi, (32, 32))
        pixels = small_roi.reshape(-1, 3)
        
        # Calcular el color promedio
        dominant_color = tuple(map(int, np.mean(pixels, axis=0)))
        return dominant_color

    def _calculate_component_hash(self, frame: np.ndarray, bbox: tuple) -> str:
        """Calcula un hash simple para el componente basado en su apariencia."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        # Redimensionar y convertir a escala de grises para el hash
        small_roi = cv2.resize(roi, (8, 8))
        gray_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2GRAY)
        
        # Crear hash basado en valores de píxeles
        hash_value = ""
        mean_value = np.mean(gray_roi)
        for pixel in gray_roi.flatten():
            hash_value += "1" if pixel > mean_value else "0"
            
        return hash_value

    def detect_components(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta componentes en un frame usando YOLO.
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            Lista de detecciones
        """
        results = self.model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            
            if conf < self.conf_threshold or int(class_id) not in self.target_classes:
                continue

            bbox = (x1, y1, x2, y2)
            component_hash = self._calculate_component_hash(frame, bbox)
            dominant_color = self._calculate_dominant_color(frame, bbox)
            
            detection = Detection(
                class_id=int(class_id),
                confidence=conf,
                bbox=bbox,
                component_hash=component_hash,
                dominant_color=dominant_color
            )
            detections.append(detection)

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Dibuja las detecciones en el frame."""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            label = f"{self.target_classes[det.class_id]} {det.confidence:.2f}"
            
            # Dibujar bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), det.dominant_color, 2)
            
            # Dibujar etiqueta
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return frame_copy 