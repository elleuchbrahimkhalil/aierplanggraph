from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def print_message() -> None:
    print("Airflow is running from Docker on Windows.")


with DAG(
    dag_id="hello_world_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["demo"],
) as dag:
    PythonOperator(
        task_id="print_message",
        python_callable=print_message,
    )
