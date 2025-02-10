# Sistema de Vigilancia Modular con YOLO11

Este sistema implementa un sistema de vigilancia modular basado en YOLO11 para la detección y seguimiento de personas y componentes.

## Características

- Detección de personas y componentes usando YOLO11
- Múltiples variantes de modelo disponibles (nano, small, medium, large, xlarge)
- Tracking basado en componentes y color
- Sistema modular y extensible
- Logging detallado de eventos
- Interfaz gráfica básica

## Requisitos

- Python 3.8+
- CUDA compatible GPU (recomendado)
- Cámara web o fuente de video

## Modelos Disponibles

El sistema soporta diferentes variantes del modelo YOLO11:

- `yolo11n.pt`: Modelo nano (más rápido, menor precisión)
- `yolo11s.pt`: Modelo small (balance velocidad/precisión)
- `yolo11m.pt`: Modelo medium (precisión mejorada)
- `yolo11l.pt`: Modelo large (alta precisión)
- `yolo11x.pt`: Modelo xlarge (máxima precisión)

Por defecto, se usa el modelo nano (`yolo11n.pt`). Puedes cambiar el modelo en `config.json`.

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd surveillance-system
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Configuración

El sistema se configura a través del archivo `config.json`. Los principales parámetros son:

```json
{
    "video": {
        "source": 0,          // ID de cámara o ruta de video
        "width": 640,         // Ancho del frame
        "height": 480         // Alto del frame
    },
    "detection": {
        "confidence_threshold": 0.5,  // Umbral de confianza YOLO
        "target_classes": [0, 24, 26, 27, 28, 32]  // Clases a detectar
    },
    "tracking": {
        "max_disappeared": 50,  // Frames máximos sin detección
        "max_distance": 50.0    // Distancia máxima para asociación
    }
}
```

## Uso

1. Ejecutar el sistema:
```bash
python main.py
```

2. Controles:
- Presionar 'q' para salir
- Los eventos se registran en la carpeta `logs/`

## Estructura del Proyecto

```
surveillance-system/
├── main.py                 # Script principal
├── video_processor.py      # Procesamiento de video
├── component_analyzer.py   # Detección con YOLO
├── component_tracker.py    # Sistema de tracking
├── data_logger.py         # Sistema de logging
├── basic_gui.py           # Interfaz gráfica
├── config.json            # Configuración
└── requirements.txt       # Dependencias
```

## Extensibilidad

El sistema está diseñado para ser modular y extensible. Algunos puntos de extensión:

- Nuevos detectores en `ComponentAnalyzer`
- Algoritmos de tracking alternativos en `ComponentTracker`
- Visualizaciones adicionales en `BasicGUI`
- Nuevos tipos de logging en `DataLogger`

## Logging

Los logs se guardan en la carpeta `logs/`:
- `surveillance.log`: Log general del sistema
- `tracking_*.csv`: Datos de tracking en formato CSV
- `events.json`: Eventos del sistema en formato JSON

## Contribuir

1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit los cambios
4. Push a la rama
5. Crear un Pull Request

## Licencia

MIT License 