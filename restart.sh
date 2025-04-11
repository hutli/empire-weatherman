#!/bin/sh
clear
docker-compose down --remove-orphans
docker-compose up --build -d
docker logs -f empire-weatherman_empire-weatherman_1
