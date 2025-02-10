import cv2
import time

def test_camera():
    print("Iniciando prueba de cámara...")
    
    # Probar cámara 0
    print("Probando cámara 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara 0")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Crear ventana
    cv2.namedWindow("Test Camera", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Leer frame
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame")
                break
            
            # Mostrar frame
            cv2.imshow("Test Camera", frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Mostrar FPS
            time.sleep(0.01)  # Pequeña pausa para no saturar el CPU
            
    except Exception as e:
        print(f"Error durante la prueba: {e}")
    
    finally:
        print("Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()
        print("Prueba finalizada")

if __name__ == "__main__":
    test_camera() 