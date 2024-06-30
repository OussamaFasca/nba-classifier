FROM python:3.11-slim

WORKDIR /app
COPY . /app/

RUN pip3 install -r requirements.txt

# Runing unit tests
RUN python3 test.py

CMD ["python3","api.py"]