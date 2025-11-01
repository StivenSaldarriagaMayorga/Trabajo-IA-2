from sklearn.metrics import classification_report, confusion_matrix
from dataset import dataframes
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

def Red_Neuronal(dataframe):

    #Tomar los datos del dataframe
    X_train, X_test, y_train, y_test = dataframe

    # --- One-hot encoding ---
    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)

    #Definir la arquitectura de la red
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='tanh'),
        Dropout(0.3),
        Dense(y_train_nn.shape[1], activation='softmax')
    ])

    #Compilar el modelo
    model.compile(optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

    #Callbacks para detener el modelo cuando deja de mejorar
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]

    #Entrenar la red neuronal
    history = model.fit(
        X_train, y_train_nn,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_nn),
        callbacks=callbacks,
        verbose=1,)

    #Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test_nn)
    print(f"Accuracy: {accuracy:.4f}")

    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Reporte de métricas
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

for dataframe in dataframes:
    Red_Neuronal(dataframe)
