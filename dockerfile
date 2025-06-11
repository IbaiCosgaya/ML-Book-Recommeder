FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias para compilar librerías como scikit-surprise
# build-essential: para compilación de C/C++
# git: por si alguna dependencia se descarga desde git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
# Utilizamos 'pip install' con '--no-cache-dir' para reducir el tamaño final de la imagen.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación al contenedor
# Esto incluye app.py, la carpeta app_collaborative, df_combined_books_final.parquet, DeLibreroo.png
COPY . .

# Exponer el puerto que usa Streamlit (esto es más informativo para Docker)
EXPOSE 7860

# Configurar Streamlit para Hugging Face Spaces (o cualquier entorno headless)
# Esto es crucial para un despliegue sin problemas en servidores
RUN mkdir -p ~/.streamlit/
RUN echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml
RUN echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = 7860\n\
" > ~/.streamlit/config.toml

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]