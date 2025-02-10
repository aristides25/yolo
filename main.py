import json
import sys
from pathlib import Path
from typing import Dict, Any

from video_processor import VideoProcessor
from component_analyzer import ComponentAnalyzer
from component_tracker import ComponentTracker
from data_logger import DataLogger
from basic_gui import BasicGUI

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
        # Inicializar procesador de video
        video_processor = VideoProcessor(
            source=config['video']['source'],
            width=config['video']['width'],
            height=config['video']['height']
        )
        
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
        
        # Inicializar GUI
        gui = BasicGUI()
        
        # Iniciar procesamiento
        with video_processor, gui:
            logger.log_event("sistema_iniciado", {"config": config})
            
            for frame in video_processor.get_frames():
                # Detectar componentes
                detections = analyzer.detect_components(frame)
                
                # Actualizar tracking
                tracked_objects = tracker.update(detections)
                
                # Registrar datos
                logger.log_tracked_objects(tracked_objects)
                
                # Actualizar visualización
                gui.update_display(frame, tracked_objects)
                
                # Salir si la GUI se cierra
                if not gui.running:
                    break
                    
    except KeyboardInterrupt:
        logger.log_event("sistema_detenido", {"reason": "keyboard_interrupt"})
    except Exception as e:
        logger.log_error(f"Error en el sistema: {str(e)}")
        raise
    finally:
        # Asegurar que todos los recursos se liberan
        logger.log_event("sistema_finalizado", {})

if __name__ == "__main__":
    main() 