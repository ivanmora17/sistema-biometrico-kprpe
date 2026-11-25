"""Crea la base de datos con las 50 personas del profesor + la nuestra (Paula Etxeberria, Twin Melody).

    data/Dataset/<nombre>_<anio>.<ext>      fotos comunes del profesor (49-50)
    data/propia/<nombre>_<anio>.<ext>       nuestra persona (1 foto)

El id es el nombre del fichero sin el anio final: 'keanu_reeves_2021.png' -> 'keanu_reeves'.

Uso:  python scripts/inscribir_dataset.py [--bd bd/base_datos.csv]
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")
from sistema_biometrico import SistemaBiometrico  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def id_desde_fichero(p: Path) -> str:
    return re.sub(r"_\d{4}$", "", p.stem)


def fotos_inscripcion():
    for carpeta in (RAIZ / "data" / "Dataset", RAIZ / "data" / "propia"):
        if carpeta.exists():
            yield from sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", default=str(RAIZ / "bd" / "base_datos.csv"))
    a = ap.parse_args()
    Path(a.bd).unlink(missing_ok=True)
    s = SistemaBiometrico(a.bd)
    for f in fotos_inscripcion():
        s.inscribir(id_desde_fichero(f), f)
        print(f"  inscrito {id_desde_fichero(f):40s} <- {f.relative_to(RAIZ)}")
    print(f"{len(s.bd.identidades())} personas inscritas en {a.bd}")
