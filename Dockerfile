# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY ./app /code/app

# Expose port 8000 (standard for FastAPI apps)
EXPOSE 8000

# Command to run the application using uvicorn
# Use 0.0.0.0 to make it accessible outside the container
# Use --reload for development (optional, remove for production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# For development with auto-reload:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

