# Playwright image includes Chromium & all deps
FROM mcr.microsoft.com/playwright/python:v1.55.0-jammy

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# Uvicorn listens on 8000
ENV PORT=8000
EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]