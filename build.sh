if [ -z "$1" ]
  then
    tag=""
else
    tag=":$1"
fi

docker build -t jordyantunes/imagine$tag .
docker push jordyantunes/imagine$tag