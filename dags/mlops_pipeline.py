from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'heart_disease_training_pipeline',
    default_args=default_args,
    description='Pipeline de entrenamiento con FeatureSpace, Postgres y MLflow',
    schedule_interval=None, # Ejecución manual por ahora
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    train_model_task = BashOperator(
        task_id='train_and_register_model',
        bash_command='python /app/scripts/train.py',
    )

    train_model_task
