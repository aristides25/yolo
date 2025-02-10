import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class PoseData:
    landmarks: np.ndarray  # 33 landmarks en formato (x, y, z)
    elbow_angle: float    # Ángulo del codo
    knee_angle: float     # Ángulo de la rodilla
    torso_angle: float    # Inclinación del torso
    posture: str         # Postura actual (DE_PIE, SENTADO, INCLINADO)
    confidence: float    # Confianza de la detección

class PoseAnalyzer:
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Inicializa el analizador de poses.
        
        Args:
            min_detection_confidence: Umbral mínimo de confianza para detecciones
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Colores por postura
        self.posture_colors = {
            "DE_PIE": (0, 255, 0),     # Verde
            "SENTADO": (0, 0, 255),     # Rojo
            "INCLINADO": (0, 165, 255)  # Naranja
        }
        
        # Historial de posturas para persistencia
        self.posture_history: Dict[int, List[str]] = {}
        self.show_angles = False  # Toggle para mostrar ángulos
        
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calcula el ángulo entre tres puntos."""
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def get_stable_posture(self, track_id: int, current_posture: str) -> str:
        """Implementa persistencia postural."""
        if track_id not in self.posture_history:
            self.posture_history[track_id] = []
            
        history = self.posture_history[track_id]
        history.append(current_posture)
        
        if len(history) > 3:
            history.pop(0)
            
        if len(history) == 3 and all(p == history[0] for p in history):
            return history[0]
        
        return history[-1] if history else current_posture
    
    def analyze_pose(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[PoseData]:
        """
        Analiza la pose en una región de interés del frame.
        
        Args:
            frame: Frame completo
            bbox: Bounding box de la persona (x1, y1, x2, y2)
            
        Returns:
            PoseData con la información de la pose o None si no se detecta
        """
        # Extraer ROI
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        # Procesar con MediaPipe
        results = self.pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
            
        # Calcular ángulos clave
        landmarks = results.pose_landmarks.landmark
        
        # Ángulo del codo (brazo derecho)
        elbow_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        # Ángulo de la rodilla (pierna derecha)
        knee_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        # Ángulo del torso
        shoulders = np.array([
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        ])
        hips = np.array([
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x +
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        ])
        torso_angle = abs(np.degrees(np.arctan2(shoulders[1] - hips[1], shoulders[0] - hips[0])))
        
        # Clasificar postura
        if knee_angle < 120:
            posture = "SENTADO"
        elif torso_angle > 15:
            posture = "INCLINADO"
        else:
            posture = "DE_PIE"
            
        # Calcular confianza promedio
        confidence = np.mean([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
        ])
        
        return PoseData(
            landmarks=np.array([[l.x, l.y, l.z] for l in landmarks]),
            elbow_angle=elbow_angle,
            knee_angle=knee_angle,
            torso_angle=torso_angle,
            posture=posture,
            confidence=confidence
        )
    
    def draw_pose(self, frame: np.ndarray, pose_data: PoseData, bbox: Tuple[float, float, float, float],
                  track_id: int) -> np.ndarray:
        """
        Dibuja el esqueleto y la información postural.
        
        Args:
            frame: Frame a dibujar
            pose_data: Datos de la pose
            bbox: Bounding box de la persona
            track_id: ID de tracking
            
        Returns:
            Frame con las visualizaciones
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.posture_colors[pose_data.posture]
        
        # Dibujar esqueleto
        landmarks = pose_data.landmarks
        connections = self.mp_pose.POSE_CONNECTIONS
        
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (
                int(x1 + landmarks[start_idx][0] * (x2 - x1)),
                int(y1 + landmarks[start_idx][1] * (y2 - y1))
            )
            end_point = (
                int(x1 + landmarks[end_idx][0] * (x2 - x1)),
                int(y1 + landmarks[end_idx][1] * (y2 - y1))
            )
            
            cv2.line(frame, start_point, end_point, color, 2)
        
        # Dibujar etiqueta
        label = f"Persona {track_id} | ID: {track_id} | POSTURE: {pose_data.posture}"
        if self.show_angles:
            label += f" | ANGLES: {int(pose_data.elbow_angle)}° (E), {int(pose_data.knee_angle)}° (K)"
        
        # Fondo para el texto
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - 25), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Texto
        cv2.putText(frame, label, (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def toggle_angles(self):
        """Alterna la visualización de ángulos."""
        self.show_angles = not self.show_angles 