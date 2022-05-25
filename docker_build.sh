sudo service docker start
sleep 1
docker build -t dc/entood:latest .
sudo service docker stop
