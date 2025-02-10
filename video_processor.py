import cv2
import threading
from typing import Generator, Optional, Tuple
import numpy as np

class VideoProcessor:
    def __init__(self, source: int = 0, width: int = 640, height: int = 480):
        """
        Inicializa el procesador de video.
        
        Args:
            source: ID de la cámara o ruta del video
            width: Ancho deseado del frame
            height: Alto deseado del frame
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self._running = False
        self._lock = threading.Lock()
        self._current_frame = None

    def start(self) -> None:
        """Inicia la captura de video en un hilo separado."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la fuente de video")
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Bucle principal de captura de frames."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self._running = False
                break
                
            with self._lock:
                self._current_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Obtiene el frame más reciente de manera thread-safe."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def get_frames(self) -> Generator[np.ndarray, None, None]:
        """Generador que produce frames continuamente."""
        while self._running:
            frame = self.get_frame()
            if frame is not None:
                yield frame

    def stop(self) -> None:
        """Detiene la captura de video y libera recursos."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join()
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 