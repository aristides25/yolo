from ultralytics import YOLO
import cv2
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import urllib.request
import os

@dataclass
class DetectedObject:
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]
    risk_level: str

class ObjectDetector:
    def __init__(self, model_path: str = 'yolo11n.pt', conf_threshold: float = 0.5):
        """
        Inicializa el detector de objetos.
        
        Args:
            model_path: Ruta al modelo YOLO11
            conf_threshold: Umbral de confianza para detecciones
        """
        # Clases de interés del dataset COCO
        self.target_classes = {
            0: {'name': 'person', 'risk': 'None', 'color': (0, 255, 0)},      # Verde
            67: {'name': 'cell phone', 'risk': 'Low', 'color': (0, 255, 0)},  # Verde
            76: {'name': 'book', 'risk': 'Low', 'color': (0, 255, 0)},        # Verde
            26: {'name': 'handbag', 'risk': 'Medium', 'color': (0, 165, 255)},# Naranja
            39: {'name': 'bottle', 'risk': 'Low', 'color': (0, 255, 0)},      # Verde
            43: {'name': 'knife', 'risk': 'High', 'color': (0, 0, 255)},      # Rojo
            75: {'name': 'gun', 'risk': 'High', 'color': (0, 0, 255)}         # Rojo
        }
        
        # URL del modelo YOLO11n
        model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        
        # Si el modelo no existe localmente, descargarlo
        if not os.path.exists(model_path):
            print(f"Descargando modelo {model_path} desde {model_url}...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Modelo descargado exitosamente en {model_path}")
            except Exception as e:
                print(f"Error al descargar el modelo: {e}")
                raise
        
        try:
            self.model = YOLO(model_path)
            print(f"Modelo {model_path} cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
            
        self.conf_threshold = conf_threshold

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[DetectedObject], Dict[str, Any]]:
        """
        Detecta objetos en un frame y genera la salida estructurada.
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            Tupla de (lista de detecciones, datos JSON)
        """
        results = self.model(frame)[0]
        detections = []
        json_data = {
            "timestamp": time.time(),
            "objects": []
        }

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            class_id = int(class_id)
            
            if conf < self.conf_threshold or class_id not in self.target_classes:
                continue

            class_info = self.target_classes[class_id]
            detected_obj = DetectedObject(
                class_name=class_info['name'],
                class_id=class_id,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                risk_level=class_info['risk']
            )
            detections.append(detected_obj)
            
            # Agregar al JSON
            json_data["objects"].append({
                "class": detected_obj.class_name,
                "confidence": detected_obj.confidence,
                "position": list(detected_obj.bbox),
                "risk_level": detected_obj.risk_level
            })

        return detections, json_data

    def draw_detections(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """
        Dibuja las detecciones en el frame.
        
        Args:
            frame: Frame de video
            detections: Lista de objetos detectados
            
        Returns:
            Frame con las detecciones dibujadas
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.target_classes[det.class_id]['color']
            
            # Dibujar bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Preparar etiquetas
            class_label = f"{det.class_name} ({det.confidence:.2f})"
            risk_label = f"Riesgo: {det.risk_level}"
            
            # Dibujar fondo para el texto
            class_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            risk_size = cv2.getTextSize(risk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            max_width = max(class_size[0], risk_size[0])
            
            cv2.rectangle(frame_copy, 
                         (x1, y1 - 40), 
                         (x1 + max_width, y1), 
                         color, -1)
            
            # Dibujar textos
            cv2.putText(frame_copy, class_label, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame_copy, risk_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_copy 