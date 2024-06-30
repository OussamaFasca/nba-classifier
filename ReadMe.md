to build the docker image :
```
docker build . -t nba_classifier
```
to run the image
```
docker run -p 5000:5000 -t nba_classifier
```
to test the api using curl :
```
curl -X POST http://0.0.0.0:5000/predict -H "Content-Type: application/json" -d "{\"features\": [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.5, 0.6, 0.4, 0.8, 0.3, 0.7, 0.2, 0.9, 0.1]}"
```