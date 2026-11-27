FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Version del extractor: webface12m (por defecto) o webface4m
ARG KPRPE=webface12m
ENV SB_KPRPE=${KPRPE}

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pesos dentro de la imagen (~470 MB, versiones fijadas y comprobadas por SHA-256):
# el contenedor funciona sin conexion.
COPY sistema_biometrico/ sistema_biometrico/
COPY scripts/ scripts/
RUN python scripts/descargar_modelos.py

# Base de datos: se genera al construir la imagen con una clave aleatoria propia de esta imagen
# (secreto/clave). Asi la BD sale firmada y con templates protegidos, y la clave no esta en git.
COPY data/ data/
COPY main.py .
RUN useradd --create-home --uid 1000 sb \
 && python scripts/inscribir_dataset.py \
 && chown -R sb:sb bd secreto \
 && chmod 700 secreto && chmod 600 secreto/clave
USER sb

# Uso:
#   docker build -t sb-kprpe .
#   docker run --rm -v "$PWD/fotos:/fotos:ro" sb-kprpe identificar /fotos/cara.jpg
#   docker run --rm -v "$PWD/fotos:/fotos:ro" sb-kprpe verificar keanu_reeves /fotos/cara.jpg
# Para que las inscripciones y el bloqueo por intentos persistan entre ejecuciones:
#   docker volume create sb-bd
#   docker run --rm -v sb-bd:/app/bd -v "$PWD/fotos:/fotos:ro" sb-kprpe inscribir yo /fotos/yo.jpg
# (un volumen con nombre vacio se rellena con la BD de la imagen la primera vez)
ENTRYPOINT ["python", "main.py"]
CMD ["listar"]
