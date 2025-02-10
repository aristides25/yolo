import json
import csv
import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from component_tracker import TrackedObject
import time

class DataLogger:
    def __init__(self, log_dir: str = "logs"):
        """
        Inicializa el sistema de logging.
        
        Args:
            log_dir: Directorio donde se guardarán los logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar logging básico
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "surveillance.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("Surveillance")
        
        # Preparar archivo CSV para tracking
        self.csv_path = self.log_dir / f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'track_id', 'class_id', 'confidence',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'component_hash', 'dominant_color'
            ])

    def log_tracked_objects(self, tracked_objects: Dict[int, TrackedObject]):
        """
        Registra los objetos tracked en el archivo CSV.
        
        Args:
            tracked_objects: Diccionario de objetos tracked
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for track_id, obj in tracked_objects.items():
                det = obj.detection
                writer.writerow([
                    timestamp,
                    track_id,
                    det.class_id,
                    det.confidence,
                    *det.bbox,
                    det.component_hash,
                    str(det.dominant_color)
                ])

    def log_event(self, event_type: str, details: dict):
        """
        Registra un evento en el log.
        
        Args:
            event_type: Tipo de evento (e.g., 'nueva_persona', 'salida_persona')
            details: Detalles adicionales del evento
        """
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            **details
        }
        
        self.logger.info(f"Evento: {event_type} - {json.dumps(details)}")
        
        # Guardar evento en archivo JSON
        events_file = self.log_dir / "events.json"
        try:
            events = []
            if events_file.exists():
                try:
                    with open(events_file, 'r') as f:
                        events = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning("Archivo de eventos corrupto, creando nuevo archivo")
                    events = []
                    
            events.append(event_data)
            
            with open(events_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error al guardar evento: {str(e)}")
            # Crear backup del archivo corrupto
            if events_file.exists():
                backup_file = self.log_dir / f"events_backup_{int(time.time())}.json"
                try:
                    events_file.rename(backup_file)
                    self.logger.info(f"Backup creado: {backup_file}")
                except Exception as be:
                    self.logger.error(f"Error al crear backup: {str(be)}")

    def log_error(self, error_msg: str, details: dict = None):
        """
        Registra un error en el log.
        
        Args:
            error_msg: Mensaje de error
            details: Detalles adicionales del error
        """
        if details:
            self.logger.error(f"{error_msg} - {json.dumps(details)}")
        else:
            self.logger.error(error_msg) 