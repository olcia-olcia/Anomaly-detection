run:
	docker run  -p 8000:8000 olcia/lp-service
build:
	docker build -t olcia/lp-service . 
inspect:
	docker exec -it 87a3 /bin/bash 
test1:
	curl -X 'POST' \
	'http://0.0.0.0:8000/prediction' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{\
	"feature_vector": [\
		1,1\
	],\
	"score": true\
	}'

test2: 
		curl -X 'POST' \
	'http://0.0.0.0:8000/prediction' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{\
	"feature_vector": [\
		100,-0.99\
	],\
	"score": true\
	}'
test3:
	curl -X 'GET' \
	'http://0.0.0.0:8000/model_information' \
	-H 'accept: application/json'