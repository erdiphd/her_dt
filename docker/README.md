# Dockerfile for Outpace


## To create docker image 
## Go previous folder then type
`docker build -t erditmp/her_dt -f docker/Dockerfile .`

## To run HER_DT docker container

For example;

` cd vlad_thesis` 

`docker container run -v ${PWD}:/home/user/her_dt/ --name her_dt --rm -it erditmp/her_dt:latest`


