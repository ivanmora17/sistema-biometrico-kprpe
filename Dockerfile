FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pesos dentro de la imagen (~470 MB): el contenedor funciona sin conexion.
COPY sistema_biometrico/ sistema_biometrico/
COPY scripts/ scripts/
RUN python scripts/descargar_modelos.py \
 && python -c "from sistema_biometrico import SistemaBiometrico; SistemaBiometrico('/tmp/x.csv')"

COPY main.py .
COPY bd/ bd/

# Uso:
#   docker build -t sb-kprpe .
#   docker run --rm -v "$PWD/fotos:/fotos" sb-kprpe identificar /fotos/cara.jpg
#   docker run --rm -it -v "$PWD/fotos:/fotos" --entrypoint python sb-kprpe   (consola Python)
ENTRYPOINT ["python", "main.py"]
CMD ["listar"]
