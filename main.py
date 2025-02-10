import json
import sys
from pathlib import Path
import cv2
import time
import numpy as np

from object_detector import ObjectDetector, DetectedObject
from component_analyzer import Detection
from component_tracker import ComponentTracker
from data_logger import DataLogger

def load_config(config_path: str = "config.json") -> dict:
    """Carga la configuración desde el archivo JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

def convert_detection(det: DetectedObject) -> Detection:
    """Convierte una detección de YOLO al formato del tracker."""
    return Detection(
        class_id=det.class_id,
        confidence=det.confidence,
        bbox=det.bbox,
        component_hash=str(hash(f"{det.bbox}{det.class_id}")),
        dominant_color=(0, 255, 0)  # Color verde por defecto
    )

def main():
    # Cargar configuración
    try:
        config = load_config()
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")
        sys.exit(1)

    # Inicializar componentes
    logger = DataLogger()
    try:
        # Inicializar detector y tracker
        detector = ObjectDetector(
            model_path=config['detection']['model'],
            conf_threshold=config['detection']['confidence_threshold']
        )
        tracker = ComponentTracker(
            max_disappeared=config['tracking']['max_disappeared'],
            max_distance=config['tracking']['max_distance']
        )

        # Inicializar captura de video
        print("Iniciando captura de video...")
        cap = cv2.VideoCapture(config['video']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['height'])

        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        # Crear ventana
        window_name = "Sistema de Vigilancia"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        logger.log_event("sistema_iniciado", {"config": config})
        print("Sistema iniciado. Presiona 'q' para salir.")

        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame")
                break

            # Detectar objetos
            detections, json_data = detector.detect_objects(frame)
            
            # Convertir detecciones al formato del tracker
            tracked_detections = []
            for det in detections:
                tracked_det = Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    component_hash=str(hash(f"{det.bbox}{det.class_id}")),
                    dominant_color=detector.target_classes[det.class_id]['color']
                )
                tracked_detections.append(tracked_det)
            
            # Actualizar tracker
            tracked_objects = tracker.update(tracked_detections)
            
            # Dibujar resultados
            display_frame = frame.copy()
            
            # Dibujar detecciones con tracking
            for track_id, tracked_obj in tracked_objects.items():
                det = tracked_obj.detection
                x1, y1, x2, y2 = map(int, det.bbox)
                
                # Color basado en el track_id y tipo de objeto
                if det.class_id == 0:  # Persona
                    color = (
                        (track_id * 50) % 255,
                        (track_id * 100) % 255,
                        (track_id * 150) % 255
                    )
                    label = f"Persona {track_id} - ID: {track_id}"
                else:
                    color = det.dominant_color
                    class_name = detector.target_classes[det.class_id]['name']
                    risk = detector.target_classes[det.class_id]['risk']
                    label = f"{class_name} - Riesgo: {risk}"
                
                # Dibujar bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Fondo para el texto
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(display_frame, 
                            (x1, y1 - 25), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Texto
                cv2.putText(display_frame, label, (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Registrar datos
            logger.log_event("detections", json_data)
            
            # Mostrar frame
            cv2.imshow(window_name, display_frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Pequeña pausa para no saturar el CPU
            time.sleep(0.01)
                    
    except KeyboardInterrupt:
        logger.log_event("sistema_detenido", {"reason": "keyboard_interrupt"})
    except Exception as e:
        logger.log_error(f"Error en el sistema: {str(e)}")
        raise
    finally:
        # Liberar recursos
        print("Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()
        logger.log_event("sistema_finalizado", {})
        print("Sistema finalizado")

if __name__ == "__main__":
    main() 