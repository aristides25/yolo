import cv2
import numpy as np
from typing import Dict, Optional
import threading
from queue import Queue
from component_tracker import TrackedObject

class BasicGUI:
    def __init__(self, window_name: str = "Surveillance System"):
        """
        Inicializa la interfaz gráfica básica.
        
        Args:
            window_name: Nombre de la ventana
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Cola para comunicación entre hilos
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.display_thread = None

    def start(self):
        """Inicia el hilo de visualización."""
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.start()

    def stop(self):
        """Detiene el hilo de visualización."""
        self.running = False
        if self.display_thread is not None:
            self.display_thread.join()
        cv2.destroyAllWindows()

    def _display_loop(self):
        """Bucle principal de visualización."""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow(self.window_name, frame)
                
                # Salir si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

    def update_display(self, frame: np.ndarray, tracked_objects: Dict[int, TrackedObject]):
        """
        Actualiza la visualización con el frame actual y los objetos tracked.
        
        Args:
            frame: Frame actual
            tracked_objects: Diccionario de objetos tracked
        """
        display_frame = frame.copy()
        
        # Dibujar objetos tracked
        for track_id, obj in tracked_objects.items():
            det = obj.detection
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Color basado en el track_id (para distinguir diferentes personas)
            color = (
                (track_id * 50) % 255,
                (track_id * 100) % 255,
                (track_id * 150) % 255
            )
            
            # Dibujar bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar ID y clase
            label = f"ID: {track_id} ({det.class_id})"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dibujar información adicional
            info_text = f"Conf: {det.confidence:.2f}"
            cv2.putText(display_frame, info_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Actualizar frame en la cola
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(display_frame)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 