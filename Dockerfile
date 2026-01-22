
FROM python:3.11-slim

# setting a working directory in the container
WORKDIR /app

# install system dependencies to handle image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# copy only requirements first to leverage caching
COPY requirements.txt .

# Install dependencies with caching
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into container
COPY . .

# expose 5000 for Flask to run
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Run the app
CMD ["python", "app.py"]