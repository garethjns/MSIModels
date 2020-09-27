FROM python:3.8-buster
EXPOSE 5000

COPY msi_models msi_models
COPY requirements.txt .
COPY tests tests
COPY run.py .

RUN pip install -r requirements.txt

RUN python -m unittest discover

RUN mkdir /data
RUN python run.py

CMD ["mlflow", "ui", "--host", "0.0.0.0"]
