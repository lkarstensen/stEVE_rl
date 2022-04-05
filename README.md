# stacieRL
 ## This branch is an example for using the docker image with MongoDB
* you can find the docker image in this [Gitlab Project](https://gitlab.cc-asp.fraunhofer.de/lnk-dr/stacie_db_server/container_registry) 
* you can find the script with a training example that shows all functions of the ReplayBuffers for the DB [here](examples/tiltmaze/db_training.py)   
* in order to use the docker image, you need to install docker on your pc
* then you need to enter following commands (maybe you must be a member of the gitlab project above to do this):
```
(sudo) docker build -t registry.gitlab.cc-asp.fraunhofer.de/lnk-dr/stacie_db_server .
(sudo) docker run -p 6666:65430 -v mongodbdata:/data/db -it registry.gitlab.cc-asp.fraunhofer.de/lnk-dr/stacie_db_server
```
* the -p and -v flags are optional: -p is for open the ports of the docker container and the -v is for using a docker volume to save your data from the database or else all data gets lost

* This [ReadMe](https://gitlab.cc-asp.fraunhofer.de/lnk-dr/stacie_db_server/-/blob/docker_db/readme.md) shows the whole communication procedure between stacierl and the docker image 




