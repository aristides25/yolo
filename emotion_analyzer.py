import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import urllib.request
import tensorflow_hub as hub

@dataclass
class EmotionData:
    emotion: str                # Emoción detectada
    confidence: float           # Confianza de la detección
    face_landmarks: np.ndarray  # 478 landmarks faciales
    bbox: Tuple[int, ...]      # Bounding box del rostro
    mesh_points: List[Tuple[float, float]]  # Puntos del mesh facial

class EmotionAnalyzer:
    def __init__(self, model_path: str = 'fer_plus.h5'):
        """
        Inicializa el analizador de emociones.
        
        Args:
            model_path: Ruta al modelo FER+ pre-entrenado
        """
        # Lista de emociones (definir antes de _download_model)
        self.emotions = ['feliz', 'triste', 'enojado', 'sorpresa', 
                        'neutral', 'asco', 'miedo']
        
        # Colores para cada emoción (BGR)
        self.emotion_colors = {
            'feliz': (0, 255, 0),     # Verde
            'triste': (255, 0, 0),    # Azul
            'enojado': (0, 0, 255),   # Rojo
            'sorpresa': (255, 255, 0), # Cyan
            'neutral': (128, 128, 128), # Gris
            'asco': (128, 0, 128),     # Púrpura
            'miedo': (0, 165, 255)     # Naranja
        }
        
        # Inicializar MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cargar modelo FER+
        self.model_path = model_path
        if not Path(model_path).exists():
            self._download_model()
        self.emotion_model = tf.keras.models.load_model(model_path)
        
    def _download_model(self):
        """Descarga el modelo de reconocimiento emocional pre-entrenado."""
        print("Cargando modelo de reconocimiento emocional...")
        try:
            # Crear un modelo base usando MobileNetV2
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(64, 64, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Crear el modelo completo
            inputs = tf.keras.Input(shape=(64, 64, 1))
            x = tf.keras.layers.Rescaling(1./255)(inputs)
            x = tf.keras.layers.Resizing(64, 64)(x)
            # Convertir 1 canal a 3 canales
            x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])
            
            # Modelo base pre-entrenado
            x = base_model(x)
            
            # Capas de clasificación
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(len(self.emotions), activation='softmax')(x)
            
            self.emotion_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compilar modelo
            self.emotion_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Guardar el modelo
            self.emotion_model.save(self.model_path)
            print("Modelo cargado y guardado exitosamente")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
            
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen facial para el modelo FER+.
        
        Args:
            face_img: Imagen del rostro
            
        Returns:
            Imagen preprocesada
        """
        # Convertir a escala de grises
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        # Redimensionar a 64x64
        face_img = cv2.resize(face_img, (64, 64))
        
        # Normalizar
        face_img = face_img.astype('float32') / 255.0
        
        # Expandir dimensiones para el modelo
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
        
    def analyze_emotion(self, frame: np.ndarray, bbox: Optional[Tuple[int, ...]] = None) -> Optional[EmotionData]:
        """
        Analiza la emoción en un frame o ROI.
        
        Args:
            frame: Frame completo o ROI
            bbox: Bounding box opcional para procesar solo una región
            
        Returns:
            Datos de la emoción detectada o None si no se detecta rostro
        """
        try:
            # Procesar ROI si se proporciona bbox
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                # Asegurar que las coordenadas estén dentro de los límites
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Agregar margen para incluir más del rostro
                margin = int((y2 - y1) * 0.2)  # 20% de la altura como margen
                y1 = max(0, y1 - margin)
                y2 = min(h, y2 + margin)
                x1 = max(0, x1 - margin)
                x2 = min(w, x2 + margin)
                
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame
                
            if roi.size == 0:
                return None
                
            # Convertir a RGB para MediaPipe
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Detectar landmarks faciales
            results = self.face_mesh.process(rgb_roi)
            if not results.multi_face_landmarks:
                return None
                
            # Obtener landmarks del primer rostro
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extraer coordenadas de landmarks
            h, w = roi.shape[:2]
            mesh_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                mesh_points.append((x, y))
                
            # Extraer ROI facial ajustado para análisis de emoción
            face_roi = self._extract_face_roi(roi, mesh_points)
            if face_roi is None or face_roi.size == 0:
                return None
                
            # Preprocesar y predecir emoción
            processed_face = self.preprocess_face(face_roi)
            emotion_probs = self.emotion_model.predict(processed_face, verbose=0)[0]
            emotion_idx = np.argmax(emotion_probs)
            
            # Ajustar las coordenadas de los mesh_points al frame original si se usó ROI
            if bbox is not None:
                mesh_points = [(x + x1, y + y1) for x, y in mesh_points]
            
            return EmotionData(
                emotion=self.emotions[emotion_idx],
                confidence=float(emotion_probs[emotion_idx]),
                face_landmarks=np.array(mesh_points),
                bbox=bbox if bbox is not None else (0, 0, w, h),
                mesh_points=mesh_points
            )
            
        except Exception as e:
            print(f"Error en analyze_emotion: {e}")
            return None
        
    def _extract_face_roi(self, frame: np.ndarray, mesh_points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Extrae la región facial usando los landmarks.
        
        Args:
            frame: Frame de entrada
            mesh_points: Lista de puntos del mesh facial
            
        Returns:
            ROI facial o None si no se puede extraer
        """
        if not mesh_points:
            return None
            
        # Obtener bounding box del rostro usando landmarks
        x_coords = [p[0] for p in mesh_points]
        y_coords = [p[1] for p in mesh_points]
        
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        # Agregar margen
        margin = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        return frame[y1:y2, x1:x2]
        
    def draw_emotion(self, frame: np.ndarray, emotion_data: EmotionData) -> np.ndarray:
        """
        Dibuja el análisis emocional en el frame.
        
        Args:
            frame: Frame a dibujar
            emotion_data: Datos de la emoción detectada
            
        Returns:
            Frame con visualizaciones
        """
        try:
            # Obtener color según emoción
            color = self.emotion_colors[emotion_data.emotion]
            
            # Dibujar bounding box del rostro
            x1, y1, x2, y2 = emotion_data.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar landmarks faciales principales (ojos, nariz, boca)
            key_points = [33, 133, 362, 263, 1, 61, 291, 199]  # Índices de puntos clave
            for idx in key_points:
                if idx < len(emotion_data.mesh_points):
                    point = emotion_data.mesh_points[idx]
                    cv2.circle(frame, point, 2, color, -1)
            
            # Dibujar etiqueta con emoción
            emotion_label = f"{emotion_data.emotion.upper()} ({emotion_data.confidence:.2f})"
            
            # Calcular posición de la etiqueta
            label_y = y1 - 10 if y1 - 10 > 20 else y1 + 30
            cv2.putText(frame, emotion_label,
                       (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Error en draw_emotion: {e}")
            return frame 