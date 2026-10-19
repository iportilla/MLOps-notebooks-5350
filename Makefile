# Docker shortcuts for the spam classifier web app (app.py).
# Run from the repository root. Type `make` or `make help` to list the targets.

IMAGE     ?= spam-lab
CONTAINER ?= spam-lab
PORT      ?= 8501
MODEL     := models/production.pkl

.DEFAULT_GOAL := help
.PHONY: help build run stop restart logs clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

build: ## Build the image (needs models/production.pkl)
	@test -f $(MODEL) || { echo "$(MODEL) is missing. Create it with: python pipeline.py data/new_week_ok.csv"; exit 1; }
	docker build -t $(IMAGE) .

run: stop ## Start the app in the background at http://localhost:8501
	docker run -d --name $(CONTAINER) -p $(PORT):8501 $(IMAGE)
	@echo "App running at http://localhost:$(PORT)  (stop it with: make stop)"

stop: ## Stop and remove the running container
	-@docker rm -f $(CONTAINER) >/dev/null 2>&1 || true

restart: build run ## Rebuild the image and start it again

logs: ## Follow the app's logs (Ctrl+C to quit)
	docker logs -f $(CONTAINER)

clean: stop ## Stop the container and delete the image
	-docker rmi $(IMAGE)
