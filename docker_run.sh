sudo service docker start
sleep 1
sudo docker run --runtime nvidia -it --rm --name entood -v $(pwd):/tf/entoof \
	-p :8888:8888 -p :6060:6060 \
	--privileged=true \
	dc/entood:latest
sudo service docker stop
