to build the docker image :
```
docker build . -t nba_classifier
```
to run the image
```
docker run -p 5000:5000 -t nba_classifier
```