SHELL := /bin/bash
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

deploy:
	docker-compose build; \
	docker-compose up -d; \

down:
	docker-compose down

add:
	git add .

commit:
	git commit -a

status:
	git status

push:
	git push origin $(BRANCH)

pull:
	git pull origin $(BRANCH)

up:
	git add .; \
	git commit -a; \
	git push origin $(BRANCH); \
