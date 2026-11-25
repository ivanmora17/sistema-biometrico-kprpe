"""Sistema de reconocimiento facial: CVLFace ViT-Base KPRPE (AdaFace, WebFace4M).

Metodos pedidos en el enunciado:

  con detector          sin detector (_sd)          sin detector ni extractor (_t)
  inscribir(id, foto)   inscribir_sd(id, foto)      inscribir_t(id, template)
  verificar(id, foto)   verificar_sd(id, foto)      verificar_t(id, template)
  identificar(foto)     identificar_sd(foto)        identificar_t(template)

Etapas:
  * DETECTOR   = YuNet. Localiza la cara mas grande y recorta un cuadrado con margen.
                 Las versiones _sd se lo saltan: asumen que `foto` YA es un recorte de la cara.
  * EXTRACTOR  = alineador DFA (5 landmarks, warp a 112x112) + ViT-KPRPE -> template de 512 floats.
                 El alineador forma parte del extractor porque KPRPE necesita los keypoints.
  * COMPARADOR = similitud coseno entre templates L2-normalizados, en [-1, 1].

`foto` puede ser una ruta, un PIL.Image o un np.ndarray RGB uint8 (HxWx3).

Valores devueltos:
  inscribir*  -> None (lanza excepcion si falla)
  verificar*  -> (aceptado: bool, score: float | None)
  identificar*-> (id | None, score: float | None)     None = "no es nadie inscrito"
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .base_datos import BaseDatos, validar_template
from .modelos import Detector, Extractor

# Umbrales (similitud coseno) elegidos con scripts/evaluar.py: el impostor mas alto entre los 50
# famosos es 0.343 y el template "media de la BD" llega a 0.341, asi que 0.40 deja margen.
UMBRAL_VERIFICACION = 0.40
UMBRAL_IDENTIFICACION = 0.40

MARGEN_RECORTE = 1.6   # lado del recorte = MARGEN * max(ancho, alto) de la caja de YuNet
TAM_RECORTE = 256      # el recorte se pasa al alineador a esta resolucion
MAX_LADO = 2048        # las imagenes mas grandes se reducen antes de procesarlas
_ID_VALIDO = re.compile(r"^[\w\-\.\(\) ]{1,100}$")


class CaraNoDetectada(ValueError):
    pass


class SistemaBiometrico:
    def __init__(self, ruta_bd: str | Path = "bd/base_datos.csv",
                 umbral_verificacion: float = UMBRAL_VERIFICACION,
                 umbral_identificacion: float = UMBRAL_IDENTIFICACION,
                 dispositivo: str | None = None):
        self.bd = BaseDatos(ruta_bd)
        self.umbral_verificacion = umbral_verificacion
        self.umbral_identificacion = umbral_identificacion
        self.detector = Detector()
        self.extractor = Extractor(dispositivo)

    # ============================================================ utilidades
    @staticmethod
    def _cargar_imagen(foto) -> np.ndarray:
        if isinstance(foto, (str, Path)):
            img = Image.open(foto)
            img = ImageOps.exif_transpose(img)  # respetar orientacion EXIF de moviles
        elif isinstance(foto, Image.Image):
            img = foto
        elif isinstance(foto, np.ndarray):
            arr = foto
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Array de imagen con forma no valida: {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr[:, :, :3])
        else:
            raise TypeError(f"Tipo de foto no soportado: {type(foto)}")
        img = img.convert("RGB")
        if min(img.size) < 16:
            raise ValueError(f"Imagen demasiado pequena: {img.size}")
        if max(img.size) > MAX_LADO:  # reducir fotos enormes (p.ej. 20 Mpx de camara)
            img.thumbnail((MAX_LADO, MAX_LADO), Image.LANCZOS)
        return np.array(img)

    @staticmethod
    def _cuadrado(rgb: np.ndarray, cx: float, cy: float, lado: float) -> np.ndarray:
        """Recorte cuadrado centrado en (cx, cy), rellenando con negro fuera de la imagen."""
        img = Image.fromarray(rgb)
        l = int(round(lado))
        x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
        lienzo = Image.new("RGB", (l, l))
        lienzo.paste(img.crop((max(x0, 0), max(y0, 0), min(x0 + l, img.width), min(y0 + l, img.height))),
                     (max(-x0, 0), max(-y0, 0)))
        return np.asarray(lienzo.resize((TAM_RECORTE, TAM_RECORTE), Image.BILINEAR))

    def _recortar_con_detector(self, rgb: np.ndarray) -> np.ndarray:
        caras = self.detector.detectar(rgb)
        if not caras:
            raise CaraNoDetectada("No se ha detectado ninguna cara")
        x, y, w, h = caras[0]["bbox"]  # la cara mas grande
        return self._cuadrado(rgb, x + w / 2, y + h / 2, MARGEN_RECORTE * max(w, h))

    def _recortar_sin_detector(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        return self._cuadrado(rgb, w / 2, h / 2, max(w, h))

    def _template(self, foto, con_detector: bool) -> np.ndarray:
        rgb = self._cargar_imagen(foto)
        recorte = self._recortar_con_detector(rgb) if con_detector else self._recortar_sin_detector(rgb)
        t, _, _ = self.extractor.extraer(recorte)
        return t

    @staticmethod
    def _validar_id(id_: str) -> str:
        if not isinstance(id_, str) or not _ID_VALIDO.match(id_):
            raise ValueError(f"Identificador no valido: {id_!r}")
        return id_

    def extraer_template(self, foto, con_detector: bool = True) -> np.ndarray:
        """Expuesto para evaluacion: devuelve el template de una foto."""
        return self._template(foto, con_detector)

    # ============================================================ nucleo (_t)
    def inscribir_t(self, id, template, reemplazar: bool = False):
        id = self._validar_id(id)
        t = validar_template(template)
        if id in self.bd and not reemplazar:
            raise ValueError(f"'{id}' ya esta inscrito (usa reemplazar=True para sobrescribir)")
        self.bd.anadir(id, t, reemplazar=reemplazar)

    def verificar_t(self, id, template):
        t = validar_template(template)
        refs = self.bd.templates_de(id) if isinstance(id, str) else np.zeros((0, 512))
        if len(refs) == 0:
            return False, None
        score = float(np.max(refs @ t))
        return score >= self.umbral_verificacion, score

    def identificar_t(self, template):
        t = validar_template(template)
        if len(self.bd) == 0:
            return None, None
        scores = self.bd.templates @ t
        i = int(np.argmax(scores))
        score = float(scores[i])
        return (self.bd.ids[i] if score >= self.umbral_identificacion else None), score

    def ranking_t(self, template, k: int = 5):
        """Top-k (id, score) - util para evaluar identificacion."""
        t = validar_template(template)
        scores = self.bd.templates @ t
        orden = np.argsort(-scores)[:k]
        return [(self.bd.ids[i], float(scores[i])) for i in orden]

    # ============================================================ con detector
    def inscribir(self, id, foto, reemplazar: bool = False):
        self.inscribir_t(id, self._template(foto, con_detector=True), reemplazar)

    def verificar(self, id, foto):
        try:
            return self.verificar_t(id, self._template(foto, con_detector=True))
        except CaraNoDetectada:
            return False, None

    def identificar(self, foto):
        try:
            return self.identificar_t(self._template(foto, con_detector=True))
        except CaraNoDetectada:
            return None, None

    # ============================================================ sin detector
    def inscribir_sd(self, id, foto, reemplazar: bool = False):
        self.inscribir_t(id, self._template(foto, con_detector=False), reemplazar)

    def verificar_sd(self, id, foto):
        return self.verificar_t(id, self._template(foto, con_detector=False))

    def identificar_sd(self, foto):
        return self.identificar_t(self._template(foto, con_detector=False))
