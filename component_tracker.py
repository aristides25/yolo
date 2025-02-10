from typing import Dict, List, Optional, Tuple
import numpy as np
from component_analyzer import Detection

class TrackedObject:
    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.detection = detection
        self.disappeared = 0
        self.component_history = [detection.component_hash]
        self.color_history = [detection.dominant_color]

    def update(self, detection: Detection):
        """Actualiza el objeto tracked con una nueva detección."""
        self.detection = detection
        self.disappeared = 0
        self.component_history.append(detection.component_hash)
        self.color_history.append(detection.dominant_color)
        if len(self.component_history) > 10:  # Mantener solo últimos 10
            self.component_history = self.component_history[-10:]
            self.color_history = self.color_history[-10:]

class ComponentTracker:
    def __init__(self, max_disappeared: int = 50, max_distance: float = 50.0):
        """
        Inicializa el tracker de componentes.
        
        Args:
            max_disappeared: Máximo número de frames que un objeto puede desaparecer
            max_distance: Distancia máxima para asociar detecciones
        """
        self.next_object_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calcula la intersección sobre unión (IoU) entre dos bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 < xi1 or yi2 < yi1:
            return 0.0

        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _calculate_similarity(self, obj1: TrackedObject, det2: Detection) -> float:
        """Calcula la similitud entre un objeto tracked y una nueva detección."""
        # IoU entre bboxes
        iou = self._calculate_iou(obj1.detection.bbox, det2.bbox)
        
        # Similitud de hash de componentes
        hash_similarity = sum(1 for a, b in zip(obj1.detection.component_hash, det2.component_hash) if a == b)
        hash_similarity /= len(obj1.detection.component_hash)
        
        # Similitud de color
        color_diff = np.mean(np.abs(np.array(obj1.detection.dominant_color) - np.array(det2.dominant_color)))
        color_similarity = 1.0 - (color_diff / 255.0)
        
        # Combinar métricas
        return 0.5 * iou + 0.25 * hash_similarity + 0.25 * color_similarity

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """
        Actualiza el estado del tracker con nuevas detecciones.
        
        Args:
            detections: Lista de nuevas detecciones
            
        Returns:
            Diccionario de objetos tracked actualizados
        """
        # Si no hay objetos tracked, inicializar con las detecciones actuales
        if len(self.tracked_objects) == 0:
            for detection in detections:
                self.tracked_objects[self.next_object_id] = TrackedObject(detection, self.next_object_id)
                self.next_object_id += 1
            return self.tracked_objects

        # Si no hay detecciones, incrementar desapariciones
        if len(detections) == 0:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id].disappeared += 1
                if self.tracked_objects[obj_id].disappeared > self.max_disappeared:
                    del self.tracked_objects[obj_id]
            return self.tracked_objects

        # Calcular matriz de similitud
        similarity_matrix = np.zeros((len(self.tracked_objects), len(detections)))
        tracked_ids = list(self.tracked_objects.keys())
        
        for i, track_id in enumerate(tracked_ids):
            for j, detection in enumerate(detections):
                similarity_matrix[i, j] = self._calculate_similarity(self.tracked_objects[track_id], detection)

        # Asociar detecciones con objetos tracked
        used_detections = set()
        used_tracks = set()

        while True:
            # Encontrar el mejor match
            if similarity_matrix.size == 0 or np.max(similarity_matrix) < 0.3:
                break

            i, j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
            if i >= len(tracked_ids) or j >= len(detections):
                break

            track_id = tracked_ids[i]
            self.tracked_objects[track_id].update(detections[j])
            
            used_detections.add(j)
            used_tracks.add(i)
            
            # Marcar como usado
            similarity_matrix[i, :] = -1
            similarity_matrix[:, j] = -1

        # Manejar objetos no matcheados
        for i, track_id in enumerate(tracked_ids):
            if i not in used_tracks:
                self.tracked_objects[track_id].disappeared += 1
                if self.tracked_objects[track_id].disappeared > self.max_disappeared:
                    del self.tracked_objects[track_id]

        # Registrar nuevas detecciones
        for j, detection in enumerate(detections):
            if j not in used_detections:
                self.tracked_objects[self.next_object_id] = TrackedObject(detection, self.next_object_id)
                self.next_object_id += 1

        return self.tracked_objects 