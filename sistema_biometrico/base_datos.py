"""Base de datos de templates en CSV, con integridad obligatoria.

Formato (una fila por inscripcion):
    id,nonce,template
    keanu_reeves,<12 bytes base64>,<AES-256-GCM(R·t) base64>

* El template que se guarda es el PROTEGIDO (R·t, ver seguridad.py), y ademas va CIFRADO con
  AES-256-GCM (clave derivada de la maestra, nonce aleatorio por fila, el id como dato asociado).
  Leer el CSV no da ningun vector, y el tag GCM detecta cualquier fila cambiada, anadida sin la
  clave o movida a otro id.
* `<bd>.sello` = version + HMAC(version, CSV completo). Detecta filas BORRADAS o duplicadas.
* La ultima version vista se guarda fuera de bd/ (`secreto/version_<bd>`, firmada): restaurar una
  copia antigua completa (CSV + sello) se detecta (rollback).

Si la firma de alguna fila o el sello no cuadran, la BD NO se carga (ErrorIntegridad): el
sistema falla cerrado en vez de funcionar con una BD manipulada.
"""
from __future__ import annotations

import base64
import csv
import errno
import hmac
import io
import os
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .seguridad import ErrorIntegridad, firmar

DIM = 512


def _aad(id_: str) -> bytes:
    return f"sistema-biometrico/template|{id_}".encode("utf-8")


def _escribir_atomico(ruta: Path, contenido: str):
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_text(contenido, encoding="utf-8", newline="")
    try:
        os.replace(tmp, ruta)  # escritura atomica
    except OSError as e:
        # El destino es un fichero montado (docker -v fichero:fichero): no se puede sustituir
        # el inodo, asi que se sobrescribe en el sitio.
        if e.errno not in (errno.EBUSY, errno.EXDEV, errno.EPERM):
            raise
        ruta.write_text(contenido, encoding="utf-8", newline="")
        tmp.unlink(missing_ok=True)


class BaseDatos:
    def __init__(self, ruta: str | Path, clave_hmac: bytes, clave_cifrado: bytes,
                 ruta_version: str | Path | None = None):
        self.ruta = Path(ruta)
        self.ruta_sello = self.ruta.with_name(self.ruta.name + ".sello")
        self.k = clave_hmac
        self._aead = AESGCM(clave_cifrado)
        # Anti-rollback: el sello lleva un numero de version que crece en cada escritura, y la
        # ultima version vista se guarda FUERA de bd/ (junto a la clave). Restaurar una copia
        # antigua completa (CSV + sello) da una version menor y se rechaza.
        self.ruta_version = Path(ruta_version) if ruta_version else None
        self.version = 0
        self.ids: list[str] = []
        self.templates = np.zeros((0, DIM), dtype=np.float32)
        if self.ruta.exists():
            self.cargar()

    # ------------------------------------------------------------------ E/S
    def cargar(self):
        contenido = self.ruta.read_text(encoding="utf-8")
        sello = self.ruta_sello.read_text().strip() if self.ruta_sello.exists() else ""
        malas = []
        ids, temps = [], []
        for n, fila in enumerate(csv.DictReader(io.StringIO(contenido, newline="")), start=2):
            try:
                id_ = fila["id"]
                nonce = base64.b64decode(fila["nonce"], validate=True)
                datos = self._aead.decrypt(nonce, base64.b64decode(fila["template"], validate=True), _aad(id_))
                t = validar_template(np.frombuffer(datos, dtype="<f4"))
            except InvalidTag:
                malas.append(f"linea {n} (id={fila.get('id')!r}): no descifra (cambiada, de otro id o sin la clave)")
                continue
            except Exception as e:  # fila corrupta o manipulada
                malas.append(f"linea {n} (id={fila.get('id')!r}): {e}")
                continue
            ids.append(id_)
            temps.append(t)
        if malas:
            raise ErrorIntegridad(f"{self.ruta} manipulada o firmada con otra clave: " + "; ".join(malas[:5])
                                  + (f" (y {len(malas) - 5} mas)" if len(malas) > 5 else ""))
        version_txt, _, mac = sello.partition(":")
        if not (version_txt.isdigit() and hmac.compare_digest(
                mac, firmar(self.k, f"sello|{version_txt}|" + contenido))):
            raise ErrorIntegridad(f"El sello de {self.ruta} no cuadra: se han borrado, duplicado o "
                                  f"reordenado filas, o falta {self.ruta_sello.name}")
        version = int(version_txt)
        vista = self._version_vista()
        if version < vista:
            raise ErrorIntegridad(f"{self.ruta} es una copia ANTIGUA (version {version} < {vista}): "
                                  f"alguien ha restaurado una BD anterior (rollback)")
        self.version = version
        self._guardar_version()
        self.ids = ids
        self.templates = np.stack(temps) if temps else np.zeros((0, DIM), dtype=np.float32)

    def guardar(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        b = io.StringIO(newline="")
        w = csv.writer(b, lineterminator="\n")
        w.writerow(["id", "nonce", "template"])
        for id_, t in zip(self.ids, self.templates):
            nonce = os.urandom(12)
            c = self._aead.encrypt(nonce, np.asarray(t, dtype="<f4").tobytes(), _aad(id_))
            w.writerow([id_, base64.b64encode(nonce).decode(), base64.b64encode(c).decode()])
        contenido = b.getvalue()
        self.version += 1
        _escribir_atomico(self.ruta, contenido)
        _escribir_atomico(self.ruta_sello,
                          f"{self.version}:" + firmar(self.k, f"sello|{self.version}|" + contenido) + "\n")
        self._guardar_version()

    def _version_vista(self) -> int:
        if not self.ruta_version or not self.ruta_version.exists():
            return 0
        txt = self.ruta_version.read_text().strip()
        version_txt, _, mac = txt.partition(":")
        if not (version_txt.isdigit() and hmac.compare_digest(mac, firmar(self.k, f"version|{version_txt}"))):
            raise ErrorIntegridad(f"{self.ruta_version} manipulado")
        return int(version_txt)

    def _guardar_version(self):
        if not self.ruta_version or self.version <= self._version_vista():
            return
        self.ruta_version.parent.mkdir(parents=True, exist_ok=True)
        _escribir_atomico(self.ruta_version, f"{self.version}:" + firmar(self.k, f"version|{self.version}") + "\n")

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
