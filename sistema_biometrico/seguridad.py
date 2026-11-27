"""Medidas de seguridad del sistema.

* Clave maestra (32 bytes). Se lee de SB_CLAVE (hex) o del fichero SB_CLAVE_FICHERO
  (por defecto `secreto/clave`, fuera de bd/ y fuera de git). Si no existe se genera.
  De ella se derivan dos subclaves independientes:
    - HMAC de cada fila de la BD y sello de la BD completa (integridad).
    - Semilla de la transformacion de templates (templates cancelables).
* Templates cancelables: en la BD no se guarda el template t sino R·t, con R una matriz
  ortogonal 512x512 derivada de la clave. Como R es ortogonal, cos(R·a, R·b) = cos(a, b): la
  precision es identica. Pero un template robado de la BD (R·t) ya no sirve para los metodos
  *_t, que reciben templates "en claro" y les aplican R: R·(R·t) no se parece a R·t.
  Cambiando la clave se revocan todos los templates (hay que reinscribir).
* Score cuantizado: los metodos publicos devuelven el score redondeado hacia abajo a
  pasos de 0.1 (y >= 0). Sin resolucion fina no hay gradiente para un ataque de hill-climbing.
* Bloqueo por intentos fallidos: tras N verificaciones fallidas seguidas contra un mismo id,
  ese id queda bloqueado un tiempo. Se cuentan tambien ids inexistentes (no se revela quien
  esta inscrito). El registro se guarda junto a la BD.

LIMITE: todo esto se ejecuta en la maquina de quien usa el sistema. Quien tenga el codigo y la
clave puede quitar cualquier medida. Protege frente a quien solo usa la API/CLI o solo tiene
acceso a la BD, no frente a quien controla el proceso (ver README).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIM = 512


class ErrorIntegridad(RuntimeError):
    """La BD se ha modificado fuera del sistema (filas anadidas, borradas o cambiadas)."""


# ---------------------------------------------------------------------------- clave
def ruta_clave() -> Path:
    return Path(os.environ.get("SB_CLAVE_FICHERO", RAIZ / "secreto" / "clave"))


def ruta_version_bd(ruta_bd) -> Path:
    """Donde se guarda la ultima version vista de una BD (anti-rollback): junto a la clave."""
    nombre = hashlib.sha256(str(Path(ruta_bd).resolve()).encode()).hexdigest()[:16]
    return Path(os.environ.get("SB_ESTADO", ruta_clave().parent)) / f"version_{nombre}"


def cargar_clave(crear: bool = True) -> bytes:
    env = os.environ.get("SB_CLAVE")
    if env:
        try:
            k = bytes.fromhex(env)
        except ValueError:
            k = hashlib.sha256(env.encode("utf-8")).digest()
        if len(k) < 16:
            raise ValueError("SB_CLAVE es demasiado corta (minimo 16 bytes / 32 caracteres hex)")
        return k
    p = ruta_clave()
    if p.exists():
        k = bytes.fromhex(p.read_text().strip())
        if len(k) < 16:
            raise ValueError(f"Clave no valida en {p}")
        return k
    if not crear:
        raise FileNotFoundError(f"No hay clave: define SB_CLAVE o crea {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    k = os.urandom(32)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(k.hex())
    warnings.warn(f"Generada una clave nueva en {p}")
    return k


def subclave(clave: bytes, uso: str) -> bytes:
    return hmac.new(clave, f"sistema-biometrico/{uso}".encode(), hashlib.sha256).digest()


def firmar(clave_hmac: bytes, datos: str) -> str:
    return hmac.new(clave_hmac, datos.encode("utf-8"), hashlib.sha256).hexdigest()


# ------------------------------------------------------------ templates cancelables
class Transformacion:
    """Rotacion ortogonal secreta R (512x512) derivada de la clave."""

    def __init__(self, clave: bytes):
        semilla = int.from_bytes(subclave(clave, "transformacion"), "big")
        g = np.random.default_rng(semilla).standard_normal((DIM, DIM))
        q, r = np.linalg.qr(g)
        self.R = (q * np.sign(np.diag(r))).astype(np.float64)  # Haar-uniforme

    def aplicar(self, t: np.ndarray) -> np.ndarray:
        v = self.R @ np.asarray(t, dtype=np.float64)
        return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------- score
def cuantizar(score: float | None, paso: float) -> float | None:
    if score is None or not paso:
        return score
    # Redondeo hacia abajo (el score mostrado nunca supera al real), con una tolerancia minima
    # para que 0.99999 (error de coma flotante) se muestre como 1.0.
    return round(max(0.0, math.floor((score + 1e-4) / paso) * paso), 6)


# ------------------------------------------------------------ bloqueo por intentos
class RegistroIntentos:
    """Cuenta verificaciones fallidas seguidas por id (guardado en JSON junto a la BD).

    Los ids se guardan como HMAC (no en claro) y se cuentan aunque no existan."""

    MAX_ENTRADAS = 10000

    def __init__(self, ruta: Path, clave_hmac: bytes, max_intentos: int, bloqueo_seg: float):
        self.ruta, self.k = Path(ruta), clave_hmac
        self.max_intentos, self.bloqueo_seg = max_intentos, bloqueo_seg

    @property
    def activo(self) -> bool:
        return self.max_intentos > 0

    def _leer(self) -> dict:
        try:
            return json.loads(self.ruta.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def _escribir(self, d: dict):
        if len(d) > self.MAX_ENTRADAS:  # quitar los mas antiguos
            d = dict(sorted(d.items(), key=lambda kv: kv[1]["t"])[-self.MAX_ENTRADAS:])
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta.write_text(json.dumps(d), encoding="utf-8")

    def _clave(self, id_) -> str:
        return firmar(self.k, f"intentos|{id_}")[:32]

    def bloqueado(self, id_) -> float:
        """Segundos de bloqueo que quedan (0 si no esta bloqueado)."""
        if not self.activo:
            return 0.0
        e = self._leer().get(self._clave(id_))
        if not e or e["fallos"] < self.max_intentos:
            return 0.0
        return max(0.0, e["t"] + self.bloqueo_seg - time.time())

    def registrar(self, id_, exito: bool):
        if not self.activo:
            return
        d, c = self._leer(), self._clave(id_)
        if exito:
            d.pop(c, None)
        else:
            e = d.get(c, {"fallos": 0, "t": 0})
            if e["fallos"] >= self.max_intentos:  # el bloqueo ya expiro: empieza de nuevo
                e = {"fallos": 0, "t": 0}
            d[c] = {"fallos": e["fallos"] + 1, "t": time.time()}
        self._escribir(d)
