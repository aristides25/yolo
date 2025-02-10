import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def create_model(input_shape=(64, 64, 1), num_classes=7):
    """Crea el modelo CNN para reconocimiento de emociones."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_fer2013():
    """Carga el dataset FER2013."""
    # Descargar dataset desde Kaggle
    # Necesitas tener kaggle.json configurado
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('msambare/fer2013', path='.', unzip=True)
    
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (64, 64))
        faces.append(face.astype('float32'))
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    
    return faces, emotions

def train_model():
    """Entrena el modelo de reconocimiento de emociones."""
    print("Cargando dataset...")
    X, y = load_fer2013()
    
    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Aumentación de datos
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda x: x/255.0
    )
    
    # Crear y compilar modelo
    print("Creando modelo...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar
    print("Iniciando entrenamiento...")
    batch_size = 64
    epochs = 50
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Guardar modelo
    print("Guardando modelo...")
    model.save('fer_plus.h5')
    print("Entrenamiento completado!")
    
    # Evaluar
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Precisión en test: {score[1]*100:.2f}%')

if __name__ == "__main__":
    train_model() 