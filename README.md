# FlowerApp

# 1.Create network
$ docker network create -d overlay --attachable flower_net

# 2.	Create services
$ docker service create --name redis-flowerapp --network flower_net redis

$ docker service create --name flowers-api --replicas 3 -p 8000:8000 --mount type=volume,source=db-flower,target=/data --network flower_net krittiya2021/flowers-api-new gunicorn --bind 0.0.0.0:8000 main:app

$ docker service create --name front-end --replicas 3 -p 80:80 --network flower_net poundps/flowerapp





Group members
Soravit varanich 6022791626
Krittiya Picha 6022791659
Sibpakonpas Thongpiam 6022792632 


