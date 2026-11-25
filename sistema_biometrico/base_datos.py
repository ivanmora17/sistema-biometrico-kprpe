"""Base de datos de templates en CSV.

Formato (una fila por inscripcion):
    id,template,firma
    keanu_reeves,"0.0123;-0.0456;...(512 valores)",<hmac-sha256 hex o vacio>

Integridad: si la variable de entorno SB_CLAVE_HMAC esta definida, cada fila se firma con
HMAC-SHA256(clave, id || template). Al cargar, las filas con firma invalida se DESCARTAN y se avisa.
Esto detecta a un atacante que escribe en el CSV sin conocer la clave, pero NO protege la
confidencialidad: quien lea el CSV obtiene los templates (ver README, analisis de ataques).
"""
from __future__ import annotations

import csv
import hmac
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np

DIM = 512


def _clave() -> bytes | None:
    k = os.environ.get("SB_CLAVE_HMAC")
    return k.encode("utf-8") if k else None


def _serializar(t: np.ndarray) -> str:
    return ";".join(f"{v:.8f}" for v in t.astype(np.float32))


def _firmar(id_: str, t_txt: str, clave: bytes) -> str:
    return hmac.new(clave, f"{id_}|{t_txt}".encode("utf-8"), hashlib.sha256).hexdigest()


class BaseDatos:
    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)
        self.ids: list[str] = []
        self.templates = np.zeros((0, DIM), dtype=np.float32)
        self.filas_rechazadas = 0
        if self.ruta.exists():
            self.cargar()

    # ------------------------------------------------------------------ E/S
    def cargar(self):
        clave = _clave()
        ids, temps, rechazadas = [], [], 0
        with open(self.ruta, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                try:
                    id_, t_txt = fila["id"], fila["template"]
                    if clave is not None:
                        firma = fila.get("firma") or ""
                        if not hmac.compare_digest(firma, _firmar(id_, t_txt, clave)):
                            raise ValueError("firma HMAC invalida")
                    t = validar_template(np.array([float(v) for v in t_txt.split(";")]))
                except Exception as e:  # fila corrupta o manipulada
                    rechazadas += 1
                    warnings.warn(f"Fila descartada de {self.ruta.name} (id={fila.get('id')!r}): {e}")
                    continue
                ids.append(id_)
                temps.append(t)
        self.ids = ids
        self.templates = np.stack(temps) if temps else np.zeros((0, DIM), dtype=np.float32)
        self.filas_rechazadas = rechazadas

    def guardar(self):
        clave = _clave()
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ruta.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "template", "firma"])
            for id_, t in zip(self.ids, self.templates):
                t_txt = _serializar(t)
                w.writerow([id_, t_txt, _firmar(id_, t_txt, clave) if clave else ""])
        os.replace(tmp, self.ruta)  # escritura atomica

    # ------------------------------------------------------------ operaciones
    def __contains__(self, id_: str) -> bool:
        return id_ in self.ids

    def __len__(self) -> int:
        return len(self.ids)

    def identidades(self) -> list[str]:
        return sorted(set(self.ids))

    def anadir(self, id_: str, t: np.ndarray, reemplazar: bool = False):
        if reemplazar:
            self.eliminar(id_)
        self.ids.append(id_)
        self.templates = np.vstack([self.templates, t[None, :]])
        self.guardar()

    def eliminar(self, id_: str):
        keep = [i for i, x in enumerate(self.ids) if x != id_]
        self.ids = [self.ids[i] for i in keep]
        self.templates = self.templates[keep]

    def templates_de(self, id_: str) -> np.ndarray:
        return self.templates[[i for i, x in enumerate(self.ids) if x == id_]]


def validar_template(t) -> np.ndarray:
    """Comprueba forma y valores y devuelve el template L2-normalizado (float32[512])."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.shape[0] != DIM:
        raise ValueError(f"El template debe tener {DIM} valores (tiene {t.shape[0]})")
    if not np.all(np.isfinite(t)):
        raise ValueError("El template contiene NaN/Inf")
    n = np.linalg.norm(t)
    if n < 1e-6:
        raise ValueError("El template es (casi) nulo")
    return (t / n).astype(np.float32)
