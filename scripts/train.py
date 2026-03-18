import os
import pandas as pd
import tensorflow as tf
import keras
from keras.utils import FeatureSpace
import mlflow
import mlflow.keras
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score

# Configuración de MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URI = "postgresql+psycopg2://postgres:postgres@postgres:5432/mlops_db"

def main():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Heart_Disease_Classification")
    except Exception as e:
        print(f"Error crítico al conectar con MLflow: {e}")
        return

    print("1. Descargando y guardando datos en Postgres...")
    engine = create_engine(DB_URI)
    
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)
    
    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    
    # Guardar en Postgres
    train_dataframe.to_sql('train_data', engine, if_exists='replace', index=False)
    val_dataframe.to_sql('test_data', engine, if_exists='replace', index=False)
    
    print("2. Leyendo datos desde Postgres...")
    train_df_db = pd.read_sql('SELECT * FROM train_data', engine)
    val_df_db = pd.read_sql('SELECT * FROM test_data', engine)

    def dataframe_to_dataset(df):
        df = df.copy()
        labels = df.pop("target")
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        return ds.shuffle(buffer_size=len(df)).batch(32)

    train_ds = dataframe_to_dataset(train_df_db)
    val_ds = dataframe_to_dataset(val_df_db)

    print("3. Configurando FeatureSpace...")
    feature_space = FeatureSpace(
        features={
            "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
            "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
            "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
            "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
            "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
            "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
            "thal": FeatureSpace.string_categorical(num_oov_indices=0),
            "age": FeatureSpace.float_discretized(num_bins=30),
            "trestbps": FeatureSpace.float_normalized(),
            "chol": FeatureSpace.float_normalized(),
            "thalach": FeatureSpace.float_normalized(),
            "oldpeak": FeatureSpace.float_normalized(),
            "slope": FeatureSpace.float_normalized(),
        },
        crosses=[
            FeatureSpace.cross(feature_names=("sex", "age"), crossing_dim=64),
            FeatureSpace.cross(feature_names=("thal", "ca"), crossing_dim=16),
        ],
        output_mode="concat",
    )

    train_ds_with_no_labels = train_ds.map(lambda x, _: x)
    feature_space.adapt(train_ds_with_no_labels)

    preprocessed_train_ds = train_ds.map(lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    preprocessed_val_ds = val_ds.map(lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    print("4. Construyendo y Entrenando Modelo...")
    dict_inputs = feature_space.get_inputs()
    encoded_features = feature_space.get_encoded_features()

    x = keras.layers.Dense(32, activation="relu")(encoded_features)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(1, activation="sigmoid")(x)

    training_model = keras.Model(inputs=encoded_features, outputs=predictions)
    training_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    with mlflow.start_run():
        history = training_model.fit(preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=0)
        
        # Loggear métricas
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        
        print("5. Creando Modelo End-to-End e Inferencia...")
        inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
        
        # Guardar modelo en formato .keras local y en MLflow
        model_path = "/app/mlruns/heart_disease_model.keras"
        inference_model.save(model_path)
        mlflow.log_artifact(model_path, "model_artifact")
        
        print(f"Entrenamiento completado. Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()
