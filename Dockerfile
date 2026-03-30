# Use the official Python 3.9 slim image
FROM python:3.9-slim

# 1. Install git (This fixes the /bin/sh: git: not found error!)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit --upgrade

# 4. Copy the rest of your files
COPY . .

# 5. Force Streamlit to use the correct port
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nport = 7860\naddress = \"0.0.0.0\"\n" > ~/.streamlit/config.toml

# 6. Expose port 7860
EXPOSE 7860

# 7. Run the app
CMD ["streamlit", "run", "app.py"]
