# Use Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 10000

# Run using Gunicorn and bind to 0.0.0.0 for Render
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
