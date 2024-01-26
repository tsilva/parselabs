# Use the official Python image based on Alpine for minimal size
FROM python:3.9-alpine

# Set the working directory to /workspace
WORKDIR /workspace

# Install Tesseract and other system dependencies
RUN apk add --no-cache \
    git \
    gettext \
    iputils \
    wget \
    curl \
    git \
    open-ssh \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    msttcorefonts-installer \
    fontconfig && \
    update-ms-fonts && \
    fc-cache -f

# Install pytesseract via pip
RUN pip install --no-cache-dir pytesseract

# Set the working directory to /workspaces/labs-parser
WORKDIR /workspaces/labs-parser

# Copy the entire project directory
COPY . .

# Install python packages
# If pytesseract is in the requirements.txt, you can remove the separate pip install command above.
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Set the default command to run when the container starts
CMD ["/bin/ash"]
