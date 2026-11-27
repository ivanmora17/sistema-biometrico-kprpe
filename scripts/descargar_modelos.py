"""Descarga los pesos (CVLFace KPRPE + DFA de HuggingFace, YuNet de opencv_zoo) y aplica el parche.

Uso:  python scripts/descargar_modelos.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import hf_hub_download  # noqa: E402

from sistema_biometrico.modelos import (DIR_PESOS, REPO_KPRPE, REPO_DFA, REVISION, SHA256,  # noqa: E402
                                         YUNET_URL)

PARCHE_ORIG = """try:
    from .rpe_ops.rpe_index import RPEIndexFunction
except ImportError:"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def comprobar(p: Path, esperado: str):
    real = sha256(p)
    if real != esperado:
        p.unlink()
        sys.exit(f"ERROR: hash de {p} no coincide ({real} != {esperado}). Fichero borrado.")


def descargar_repo(repo_id: str) -> Path:
    destino = DIR_PESOS / repo_id.split("/")[1]
    destino.mkdir(parents=True, exist_ok=True)
    rev = REVISION[repo_id]
    hf_hub_download(repo_id, "files.txt", local_dir=destino, revision=rev)
    ficheros = [f for f in (destino / "files.txt").read_text().split("\n") if f]
    for f in ficheros + ["config.json", "wrapper.py", "model.safetensors"]:
        if not (destino / f).exists():
            print("  descargando", repo_id, f)
            hf_hub_download(repo_id, f, local_dir=destino, revision=rev)
    comprobar(destino / "model.safetensors", SHA256[repo_id])
    return destino


def parchear_rpe(destino: Path):
    """El codigo original intenta compilar una extension C++ con `setup.py install --user`
    y, si falla, deja cambiado el directorio de trabajo (rompe la carga de pesos).
    Lo sustituimos por: usar la extension si existe y si no, la implementacion en PyTorch puro."""
    p = destino / "models" / "vit_kprpe" / "RPE" / "__init__.py"
    s = p.read_text(encoding="utf-8")
    if "[PATCH practica]" in s:
        return
    ini, fin = s.index(PARCHE_ORIG), s.index("def build_rpe")
    nuevo = ("# [PATCH practica] No se compila `rpe_ops`; se usa la version PyTorch pura (inferencia).\n"
             "try:\n    from .rpe_ops.rpe_index import RPEIndexFunction\n"
             "except Exception:\n    RPEIndexFunction = None\n\n\n")
    p.write_text(s[:ini] + nuevo + s[fin:], encoding="utf-8")
    print("  parche aplicado a", p)


if __name__ == "__main__":
    print("Pesos en", DIR_PESOS)
    parchear_rpe(descargar_repo(REPO_KPRPE))
    descargar_repo(REPO_DFA)
    yunet = DIR_PESOS / "yunet" / "face_detection_yunet_2023mar.onnx"
    if not yunet.exists():
        yunet.parent.mkdir(parents=True, exist_ok=True)
        print("  descargando YuNet")
        urllib.request.urlretrieve(YUNET_URL, yunet)
    comprobar(yunet, SHA256["yunet"])
    print("OK")
