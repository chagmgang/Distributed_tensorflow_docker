## How to run distributed tensorflow with docker Port Communication

### First make docker images from Dockerfile

* Command
```
docker build -t test:latest .
```

* Result
```
(base) chageumgang-ui-MacBookPro:test chageumgang$ docker build -t test:latest .
Sending build context to Docker daemon   7.68kB
Step 1/2 : FROM nvidia/cuda:10.0-base-ubuntu16.04
10.0-base-ubuntu16.04: Pulling from nvidia/cuda
976a760c94fc: Pull complete
...
4189f57a58ef: Pull complete 
Digest: sha256:459a4ec73aa1a18b38838a9207f8c438f264199f5f132c8968fe6fdaa578d4cb
Status: Downloaded newer image for nvidia/cuda:10.0-base-ubuntu16.04
 ---> 723856171922
Step 2/2 : FROM tensorflow/tensorflow:1.14.0-gpu-py3
1.14.0-gpu-py3: Pulling from tensorflow/tensorflow
6abc03819f3e: Pull complete 
05731e63f211: Pull complete
... 
f401bdaa92ad: Pull complete
6669e38ab1ba: Pull complete 
5b6ac7f35d3d: Pull complete 
Digest: sha256:e72e66b3dcb9c9e8f4e5703965ae1466b23fe8cad59e1c92c6e9fa58f8d81dc8
Status: Downloaded newer image for tensorflow/tensorflow:1.14.0-gpu-py3
 ---> a7a1861d2150
Successfully built a7a1861d2150
Successfully tagged test:latest
```

### Run Container by previous image

* Command
```
(base) chageumgang-ui-MacBookPro:test chageumgang$ docker run -it --rm -d --volume=$(pwd):/app/ test:latest /bin/bash
```

* Figure out the container id
```
abbe01ccdc06df6f501a27235319c966188ddc42163eb485d0c589e73f9b7c30
```

* exec to generated container

```
docker exec -it 7c964577cb3d python train.py --job_name=learner
docker exec -it 7c964577cb3d python train.py --job_name=actor
```
