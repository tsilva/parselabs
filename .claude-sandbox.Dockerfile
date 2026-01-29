FROM claude-sandbox:latest
USER root
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*
USER claude
