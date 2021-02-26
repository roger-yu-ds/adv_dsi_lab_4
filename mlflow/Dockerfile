FROM python:3.7.0

RUN pip install --upgrade pip

RUN pip install mlflow==1.13

RUN pip install psycopg2-binary==2.8.5

RUN mkdir /mlflow/

RUN mkdir /mlflow/mlstore/

RUN mkdir /mlflow/mlruns/

CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "/mlflow/mlstore", "--default-artifact-root" , "/mlflow/mlruns"]