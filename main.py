import json
import sys
from pathlib import Path
from typing import Dict, Any
import cv2
import time
import numpy as np

from video_processor import VideoProcessor
from component_analyzer import ComponentAnalyzer
from component_tracker import ComponentTracker
from data_logger import DataLogger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Carga la configuración desde el archivo JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

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
        # Inicializar analizador de componentes
        analyzer = ComponentAnalyzer(
            model_path=config['detection']['model'],
            conf_threshold=config['detection']['confidence_threshold']
        )
        
        # Inicializar tracker
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
        window_name = "Surveillance System"
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

            # Detectar componentes
            detections = analyzer.detect_components(frame)
            
            # Actualizar tracking
            tracked_objects = tracker.update(detections)
            
            # Registrar datos
            logger.log_tracked_objects(tracked_objects)
            
            # Dibujar resultados
            display_frame = frame.copy()
            for track_id, obj in tracked_objects.items():
                det = obj.detection
                x1, y1, x2, y2 = map(int, det.bbox)
                
                # Color basado en el track_id
                color = (
                    (track_id * 50) % 255,
                    (track_id * 100) % 255,
                    (track_id * 150) % 255
                )
                
                # Dibujar bbox y etiquetas
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"Persona {track_id} - ID: {track_id} ({det.class_id})"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                info_text = f"Conf: {det.confidence:.2f}"
                cv2.putText(display_frame, info_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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