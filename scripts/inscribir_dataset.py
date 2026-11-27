"""Crea la base de datos con las fotos del profesor (49) + la nuestra (Paula Etxeberria, Twin Melody).

    data/Dataset/<nombre>_<anio>.<ext>      fotos comunes del profesor (49-50)
    data/propia/<nombre>_<anio>.<ext>       nuestra persona (1 foto)
    data/extra/<id>/*.<ext>                 fotos ADICIONALES de esa persona (se inscriben junto a la
                                            principal). Se usa para las gemelas: con 3-4 fotos de cada una
                                            el sistema las separa mucho mejor que con una sola.

El id es el nombre del fichero sin el anio final: 'keanu_reeves_2021.png' -> 'keanu_reeves'.

Cada persona se inscribe con la API publica (`inscribir(id, [fotos])`), con la misma politica que
cualquier usuario: una foto basta salvo que la cara se parezca a alguien ya inscrito; entonces hacen
falta >= 3 fotos que el sistema distinga de esa persona (es lo que pasa con Paula frente a Aitana).
La BD queda cifrada y firmada con la clave del sistema (SB_CLAVE o secreto/clave; si no existe se genera).

Uso:  python scripts/inscribir_dataset.py [--bd bd/base_datos.csv]
"""
import argparse
import os
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
    for sufijo in ("", ".sello"):
        Path(a.bd + sufijo).unlink(missing_ok=True)
    Path(a.bd).with_name(Path(a.bd).stem + ".intentos.json").unlink(missing_ok=True)
    from sistema_biometrico.seguridad import ruta_version_bd
    ruta_version_bd(a.bd).unlink(missing_ok=True)  # BD nueva: empieza la cuenta de versiones
    s = SistemaBiometrico(a.bd)
    extra = RAIZ / "data" / "extra"
    principales = {id_desde_fichero(f): f for f in fotos_inscripcion()}
    for d in (sorted(x for x in extra.iterdir() if x.is_dir()) if extra.exists() else []):
        if d.name not in principales:
            sys.exit(f"data/extra/{d.name}: no hay foto principal de esa persona")
    for id_, f in principales.items():
        d = extra / id_
        fotos = [f] + (sorted(x for x in d.iterdir() if x.suffix.lower() in EXT) if d.exists() else [])
        s.inscribir(id_, fotos, clave_admin=os.environ.get("SB_CLAVE_ADMIN"))
        print(f"  inscrito {id_:40s} <- " + ", ".join(str(x.relative_to(RAIZ)) for x in fotos))
    print(f"{len(s.bd.identidades())} personas inscritas ({len(s.bd)} templates) en {a.bd}")
