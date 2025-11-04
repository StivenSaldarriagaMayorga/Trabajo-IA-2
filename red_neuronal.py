from dataset import calcular_datos_curvas_caso8, calcular_metricas, dataframes, probar_modelo, preprocesadores
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

datos_curvas_caso8_nn = {}
def Red_Neuronal(idx_dataframe, dataframe):
    #Tomar los datos del dataframe
    X_train, X_test, y_train, y_test = dataframe

    #One-hot encoding
    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)

    #Data Augmentation
    noise_factor = 0.01
    X_train_augmented = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
    y_train_combined = np.concatenate((y_train_nn, y_train_nn), axis=0)

    #Definir la arquitectura de la red
    model = Sequential([
        Input(shape=(X_train.shape[1],)),

        # Capa de ruido gaussiano para regularización
        GaussianNoise(0.05),

        Dense(128, activation='relu', 
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='tanh', 
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', 
              kernel_regularizer=regularizers.l1(1e-5)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(y_train_nn.shape[1], activation='softmax')
    ])

    #Compilar el modelo
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #Callback para detener el modelo cuando deja de mejorar
    callback = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    #Entrenar la red neuronal
    history = model.fit(
        X_train_combined, y_train_combined,
        epochs=80,
        batch_size=32,
        validation_data=(X_test, y_test_nn),
        callbacks=[callback],
        verbose=1
    )

    #Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test_nn)

    #Métricas adicionales
    y_pred_nn = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_nn, axis=1)
    y_true_classes = np.argmax(y_test_nn, axis=1)

    metricas = calcular_metricas(y_true_classes, y_pred_classes)
    print(metricas)
    print("-"*50)

    pruebas = probar_modelo(model, preprocesadores[idx_dataframe])

    if i == 7:
        calcular_datos_curvas_caso8(datos_curvas_caso8_nn, model, X_test, y_test)

    return metricas, pruebas

metricas_nn = []
pruebas_nn = []
for i, dataframe in enumerate(dataframes):
    metricas, pruebas = Red_Neuronal(i, dataframe)
    metricas_nn.append(metricas)
    pruebas_nn.append(pruebas)
