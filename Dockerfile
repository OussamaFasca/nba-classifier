FROM python:3.11-slim

WORKDIR /app
COPY . /app/

RUN pip3 install -r requirements.txt

EXPOSE 5000

# model building
RUN python3 test.py

CMD ["python3","api.py"]