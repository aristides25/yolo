import cv2
import threading
from typing import Generator, Optional, Tuple
import numpy as np
import time

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
        print(f"Intentando abrir la fuente de video: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        # Configurar propiedades de la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verificar si la cámara se abrió correctamente
        if not self.cap.isOpened():
            available_cameras = self._list_available_cameras()
            raise RuntimeError(f"No se pudo abrir la fuente de video {self.source}. Cámaras disponibles: {available_cameras}")
        
        # Mostrar información de la cámara
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Cámara inicializada:")
        print(f"- Resolución: {actual_width}x{actual_height}")
        print(f"- FPS: {fps}")
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.start()
        print("Hilo de captura iniciado")

    def _list_available_cameras(self) -> list:
        """Lista las cámaras disponibles en el sistema."""
        available_cameras = []
        for i in range(10):  # Probar los primeros 10 índices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def _capture_loop(self) -> None:
        """Bucle principal de captura de frames."""
        frames_processed = 0
        start_time = time.time()
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al leer frame de la cámara")
                self._running = False
                break
                
            with self._lock:
                self._current_frame = frame
            
            # Calcular FPS cada 30 frames
            frames_processed += 1
            if frames_processed % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS actuales: {fps:.2f}")
                start_time = time.time()

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
            else:
                print("Frame nulo recibido")
                time.sleep(0.1)  # Pequeña pausa para no saturar el CPU

    def stop(self) -> None:
        """Detiene la captura de video y libera recursos."""
        print("Deteniendo captura de video...")
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join()
        if self.cap is not None:
            self.cap.release()
        print("Captura de video detenida")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 