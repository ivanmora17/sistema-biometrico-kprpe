"""Sistema de reconocimiento facial: CVLFace ViT-Base KPRPE (AdaFace, WebFace12M por defecto).

Metodos pedidos en el enunciado:

  con detector          sin detector (_sd)          sin detector ni extractor (_t)
  inscribir(id, foto)   inscribir_sd(id, foto)      inscribir_t(id, template)
  verificar(id, foto)   verificar_sd(id, foto)      verificar_t(id, template)
  identificar(foto)     identificar_sd(foto)        identificar_t(template)

Etapas:
  * DETECTOR   = YuNet. Localiza la cara principal y recorta un cuadrado con margen.
                 Las versiones _sd se lo saltan: asumen que `foto` YA es un recorte de la cara.
  * EXTRACTOR  = alineador DFA (5 landmarks, warp a 112x112) + ViT-KPRPE -> template de 512 floats.
                 El alineador forma parte del extractor porque KPRPE necesita los keypoints.
  * COMPARADOR = similitud coseno entre templates L2-normalizados, en [-1, 1].

`foto` puede ser una ruta, un PIL.Image o un np.ndarray RGB (uint8, o float en [0, 1]).
`template` es el que devuelve `extraer_template` / `main.py template` (512 floats, en claro).

Valores devueltos:
  inscribir*  -> None (lanza excepcion si falla)
  verificar*  -> (aceptado: bool, score: float | None)     score cuantizado a pasos de 0.1
  identificar*-> (id | None, score: float | None)          None = "no es nadie inscrito"
  score None  = la foto no tiene una cara utilizable (ninguna, varias, o de mala calidad).

Medidas de seguridad (detalle en seguridad.py y en el README):
  1. BD cifrada y autenticada (AES-256-GCM por fila + sello HMAC global); si se manipula, el
     sistema no arranca.
  2. Templates cancelables en la BD: un template robado no sirve para *_t.
  3. Score cuantizado (0.1) y >= 0: sin senal para hill-climbing; ids inexistentes dan 0.0.
  4. Bloqueo de un id tras 5 verificaciones fallidas seguidas (5 min).
  5. Coherencia 1:1 / 1:N: se rechaza si otra identidad inscrita se parece tanto o mas
     (margen 0.03) que la reclamada. Mitiga gemelos y parecidos.
  6. Control de calidad: una sola cara clara, de al menos 16 px, y alineacion fiable.
  7. Inscripcion: una persona normal se inscribe con una foto; si se parece a alguien ya
     inscrito (gemelo o doble), hacen falta >= 3 fotos que el sistema pueda distinguir de esa
     persona. Reemplazar o anadir fotos a una identidad exige la clave de administrador.
"""
from __future__ import annotations

import hmac
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .base_datos import BaseDatos, validar_template
from .modelos import Detector, Extractor
from .seguridad import (RegistroIntentos, Transformacion, cargar_clave, cuantizar, ruta_version_bd,
                        subclave)

# Umbrales (similitud coseno) elegidos con scripts/evaluar.py (extractor WebFace12M): el impostor
# mas alto entre los 50 inscritos (sin contar las gemelas) es 0.266 y el template "media de la BD"
# llega a 0.279, asi que 0.40 deja margen. Las degradaciones razonables de una foto genuina se
# quedan muy por encima (ver resultados/informe.md).
UMBRAL_VERIFICACION = 0.40
UMBRAL_IDENTIFICACION = 0.40
MIN_FOTOS_PARECIDOS = 3     # fotos minimas para inscribir a un gemelo / doble de alguien inscrito
MAX_FOTOS_INSCRIPCION = 20
MARGEN_AMBIGUEDAD = 0.03   # distancia minima entre la identidad elegida y la siguiente (si esta supera el umbral)
PASO_SCORE = 0.1           # resolucion del score que ve el usuario
MAX_INTENTOS = int(os.environ.get("SB_MAX_INTENTOS", 5))
BLOQUEO_SEG = float(os.environ.get("SB_BLOQUEO_SEG", 300))

MARGEN_RECORTE = 1.6   # lado del recorte = MARGEN * max(ancho, alto) de la caja de YuNet
TAM_RECORTE = 256      # el recorte se pasa al alineador a esta resolucion
MAX_LADO = 2048        # las imagenes mas grandes se reducen antes de procesarlas
MAX_PIXELES = 40_000_000
CONF_ORIENTACION = 0.85  # con esta confianza en la orientacion original no se prueban otras
CONF_DETECCION = 0.8   # confianza de YuNet para considerar que una cara es "clara"
RATIO_SEGUNDA_CARA = 0.5  # si hay otra cara clara de al menos este % del area -> foto ambigua
LADO_MIN_CARA = 16     # px (en la imagen tras reducirla a MAX_LADO)
LADO_MIN_DETECCION = 640  # las imagenes mas pequenas se amplian a esto antes de detectar
SCORE_MIN_ALINEADOR = 0.5
_ID_VALIDO = re.compile(r"^[\w\-\.\(\) ]{1,100}$")

Image.MAX_IMAGE_PIXELS = MAX_PIXELES  # evita "bombas de descompresion"


class CaraNoDetectada(ValueError):
    pass


class CaraNoValida(CaraNoDetectada):
    """Hay cara, pero no sirve: varias caras, demasiado pequena o mal alineada."""


class SistemaBiometrico:
    def __init__(self, ruta_bd: str | Path = "bd/base_datos.csv",
                 umbral_verificacion: float = UMBRAL_VERIFICACION,
                 umbral_identificacion: float = UMBRAL_IDENTIFICACION,
                 dispositivo: str | None = None, *,
                 paso_score: float = PASO_SCORE,
                 margen_ambiguedad: float = MARGEN_AMBIGUEDAD,
                 max_intentos: int = MAX_INTENTOS,
                 bloqueo_seg: float = BLOQUEO_SEG,
                 clave: bytes | None = None):
        clave = clave or cargar_clave()
        self._kh = subclave(clave, "hmac")
        self._transf = Transformacion(clave)
        ruta_bd = Path(ruta_bd)
        self._kc = subclave(clave, "cifrado")
        self.bd = BaseDatos(ruta_bd, self._kh, self._kc, ruta_version=ruta_version_bd(ruta_bd))
        self.intentos = RegistroIntentos(ruta_bd.with_name(ruta_bd.stem + ".intentos.json"),
                                         self._kh, max_intentos, bloqueo_seg)
        self.umbral_verificacion = umbral_verificacion
        self.umbral_identificacion = umbral_identificacion
        self.paso_score = paso_score
        self.margen = margen_ambiguedad
        self.detector = Detector()
        self.extractor = Extractor(dispositivo)

    # ============================================================ utilidades
    @staticmethod
    def _cargar_imagen(foto) -> np.ndarray:
        try:
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
                if arr.size > MAX_PIXELES * 4:
                    raise ValueError("Imagen demasiado grande")
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float64)
                    if not np.all(np.isfinite(arr)):
                        raise ValueError("La imagen contiene NaN/Inf")
                    if arr.max(initial=0) <= 1.0:  # float en [0, 1]
                        arr = arr * 255.0
                    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
                img = Image.fromarray(arr[:, :, :3])
            else:
                raise TypeError(f"Tipo de foto no soportado: {type(foto)}")
            img = img.convert("RGB")
        except Image.DecompressionBombError as e:
            raise ValueError(f"Imagen demasiado grande: {e}") from None
        if min(img.size) < 16:
            raise ValueError(f"Imagen demasiado pequena: {img.size}")
        if max(img.size) > MAX_LADO:  # reducir fotos enormes (p.ej. 20 Mpx de camara)
            img.thumbnail((MAX_LADO, MAX_LADO), Image.LANCZOS)
        return np.array(img)

    @staticmethod
    def _cuadrado(rgb: np.ndarray, cx: float, cy: float, lado: float) -> np.ndarray:
        """Recorte cuadrado centrado en (cx, cy), rellenando con negro fuera de la imagen."""
        img = Image.fromarray(rgb)
        l = max(1, int(round(lado)))
        x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
        lienzo = Image.new("RGB", (l, l))
        lienzo.paste(img.crop((max(x0, 0), max(y0, 0), min(x0 + l, img.width), min(y0 + l, img.height))),
                     (max(-x0, 0), max(-y0, 0)))
        return np.asarray(lienzo.resize((TAM_RECORTE, TAM_RECORTE), Image.BILINEAR))

    def _recortar_con_detector(self, rgb: np.ndarray) -> np.ndarray:
        """Recorte de la cara principal en la orientacion correcta.

        Las fotos de movil sin EXIF pueden venir giradas. YuNet encuentra caras tumbadas, pero con
        menos confianza (y con landmarks inventados), asi que se prueban las 4 orientaciones y se
        usa la que da la cara con mayor confianza."""
        mejor_k, mejor = None, -1.0
        for k in (0, 1, 3, 2):
            caras = self._detectar(np.ascontiguousarray(np.rot90(rgb, k)))[0]
            sc = caras[0]["score"] if caras else -1.0
            if k == 0 and sc >= CONF_ORIENTACION:
                break  # la foto ya esta derecha: no se buscan otras orientaciones
            # Otra orientacion solo gana si es claramente mejor que la original.
            if sc > mejor + (0.05 if mejor_k == 0 else 0.0):
                mejor_k, mejor = k, sc
        else:
            if mejor_k is None or mejor < 0:
                raise CaraNoDetectada("No se ha detectado ninguna cara")
            return self._recortar_orientacion(np.ascontiguousarray(np.rot90(rgb, mejor_k)))[0]
        return self._recortar_orientacion(rgb)[0]

    def _detectar(self, rgb: np.ndarray):
        # Las fotos pequenas se amplian (bicubica) antes de detectar: YuNet no ve caras diminutas.
        h, w = rgb.shape[:2]
        f = max(1.0, LADO_MIN_DETECCION / max(h, w))
        if f > 1.0:
            rgb = np.asarray(Image.fromarray(rgb).resize((round(w * f), round(h * f)), Image.BICUBIC))
        return self.detector.detectar(rgb), rgb, f

    def _recortar_orientacion(self, rgb: np.ndarray):
        caras, rgb, f = self._detectar(rgb)
        if not caras:
            raise CaraNoDetectada("No se ha detectado ninguna cara")
        claras = [c for c in caras if c["score"] >= CONF_DETECCION]
        if len(claras) >= 2:
            areas = sorted((c["bbox"][2] * c["bbox"][3] for c in claras), reverse=True)
            if areas[1] >= RATIO_SEGUNDA_CARA * areas[0]:
                raise CaraNoValida("Hay varias caras de tamano parecido: no se sabe cual es la persona")
        x, y, w, h = caras[0]["bbox"]  # la cara principal (clara y mas grande)
        if max(w, h) / f < LADO_MIN_CARA:  # tamano en la imagen original
            raise CaraNoValida(f"Cara demasiado pequena ({max(w, h) / f:.0f} px < {LADO_MIN_CARA})")
        return self._cuadrado(rgb, x + w / 2, y + h / 2, MARGEN_RECORTE * max(w, h)), caras[0]

    def _recortar_sin_detector(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        return self._cuadrado(rgb, w / 2, h / 2, max(w, h))

    def _template(self, foto, con_detector: bool) -> np.ndarray:
        rgb = self._cargar_imagen(foto)
        recorte = self._recortar_con_detector(rgb) if con_detector else self._recortar_sin_detector(rgb)
        t, score_alin, _ = self.extractor.extraer(recorte)
        if score_alin < SCORE_MIN_ALINEADOR:
            raise CaraNoValida(f"Alineacion poco fiable ({score_alin:.2f}): la imagen no parece una cara"
                               + ("" if con_detector else " centrada que ocupe el recorte"))
        return t

    @staticmethod
    def _validar_id(id_: str) -> str:
        if not isinstance(id_, str) or not _ID_VALIDO.match(id_):
            raise ValueError(f"Identificador no valido: {id_!r}")
        return id_

    def extraer_template(self, foto, con_detector: bool = True) -> np.ndarray:
        """Template (en claro) de una foto: la entrada de los metodos *_t."""
        return self._template(foto, con_detector)

    # ======================================================= comparador interno
    # Devuelven scores sin cuantizar. Son para evaluar el sistema, no forman parte de la API.
    def _scores(self, template) -> np.ndarray:
        """Similitud del template (en claro) con cada fila de la BD."""
        return self.bd.templates @ self._transf.aplicar(validar_template(template))

    def _por_identidad(self, scores: np.ndarray) -> dict[str, float]:
        mejor: dict[str, float] = {}
        for i, s in zip(self.bd.ids, scores.tolist()):
            mejor[i] = max(s, mejor.get(i, -1.0))
        return mejor

    def _verificar_raw(self, id_, template):
        por_id = self._por_identidad(self._scores(template))
        if id_ not in por_id:
            return False, None
        score = por_id[id_]
        otra = max((s for i, s in por_id.items() if i != id_), default=-1.0)
        ambigua = otra >= self.umbral_verificacion and score - otra < self.margen
        return score >= self.umbral_verificacion and not ambigua, score

    def _identificar_raw(self, template):
        if len(self.bd) == 0:
            return None, None
        orden = sorted(self._por_identidad(self._scores(template)).items(), key=lambda kv: -kv[1])
        id_, score = orden[0]
        segunda = orden[1][1] if len(orden) > 1 else -1.0
        ambigua = segunda >= self.umbral_identificacion and score - segunda < self.margen
        return (id_ if score >= self.umbral_identificacion and not ambigua else None), score

    def _ranking(self, template, k: int = 5):
        return sorted(self._por_identidad(self._scores(template)).items(), key=lambda kv: -kv[1])[:k]

    # ============================================================ nucleo (_t)
    @staticmethod
    def _lista_templates(template) -> list[np.ndarray]:
        """Un template (512 valores) o varios (lista de templates / matriz Nx512)."""
        arr = np.asarray(template, dtype=np.float64)
        filas = arr if arr.ndim == 2 else arr.reshape(1, -1)
        if len(filas) == 0 or len(filas) > MAX_FOTOS_INSCRIPCION:
            raise ValueError(f"Hay que inscribir entre 1 y {MAX_FOTOS_INSCRIPCION} templates")
        return [validar_template(f) for f in filas]

    def _comprobar_parecidos(self, id_, ts: list[np.ndarray]):
        """Politica de inscripcion frente a gemelos y dobles.

        * Las fotos de una misma inscripcion deben ser de la misma persona (similitud >= umbral).
        * Si la cara se parece a una persona ya inscrita (score >= umbral), una foto no basta: hacen
          falta al menos MIN_FOTOS_PARECIDOS, y en al menos 2/3 de ellas la cara tiene que parecerse
          mas a sus otras fotos que a esa persona (por el margen de ambiguedad). Si no, el sistema no
          podria distinguirlas y se rechaza. Esto tambien frena a quien intenta inscribirse con fotos
          de otra persona inscrita: esas fotos se parecen tanto o mas a la victima que entre si.
        No se dice con quien se confunde (no se revela quien esta inscrito)."""
        if len(ts) >= 2:
            G = np.array([[a @ b for b in ts] for a in ts])
            np.fill_diagonal(G, -1.0)
            if G.max(axis=1).min() < self.umbral_verificacion:
                raise ValueError("Las fotos de la inscripcion no parecen de la misma persona")
        otros = []
        for t in ts:
            por_id = self._por_identidad(self._scores(t))
            otros.append(max((v for k, v in por_id.items() if k != id_), default=-1.0))
        if max(otros) < self.umbral_verificacion:
            return False  # no se parece a nadie: basta con una foto
        if len(ts) < MIN_FOTOS_PARECIDOS:
            raise ValueError(f"Esa cara se parece mucho a una persona ya inscrita (gemelo o doble): para "
                             f"inscribirla hacen falta al menos {MIN_FOTOS_PARECIDOS} fotos distintas suyas")
        separables = 0
        for i, t in enumerate(ts):
            propio = max(t @ u for j, u in enumerate(ts) if j != i)
            separables += propio - otros[i] >= self.margen
        if separables < math.ceil(2 * len(ts) / 3):
            raise ValueError("Esa cara no se distingue de una persona ya inscrita ni con estas fotos: "
                             "aporta fotos mas claras o mas variadas")
        return True

    def inscribir_t(self, id, template, reemplazar: bool = False, clave_admin: str | None = None,
                    anadir: bool = False):
        """Inscribe `id` con uno o varios templates.

        reemplazar=True sustituye la identidad; anadir=True le anade templates. Las dos cosas modifican
        una identidad existente y exigen la clave de administrador (SB_CLAVE_ADMIN)."""
        id = self._validar_id(id)
        ts = self._lista_templates(template)
        admin = os.environ.get("SB_CLAVE_ADMIN")
        es_admin = bool(admin) and clave_admin is not None and hmac.compare_digest(
            clave_admin.encode(), admin.encode())
        if admin and not es_admin:
            raise PermissionError("Inscribir requiere la clave de administrador (SB_CLAVE_ADMIN)")
        if id in self.bd:
            if not (reemplazar or anadir):
                raise ValueError(f"'{id}' ya esta inscrito")
            if not es_admin:
                raise PermissionError("Modificar una identidad inscrita (reemplazar o anadir fotos) requiere "
                                      "la clave de administrador (define SB_CLAVE_ADMIN y pasa clave_admin / --admin)")
        elif anadir:
            raise ValueError(f"'{id}' no esta inscrito")
        if not anadir:
            parecido = self._comprobar_parecidos(id, ts)
            if parecido:
                warnings.warn("Inscrita una cara muy parecida a otra persona ya inscrita. Conviene que esa "
                              "persona tenga tambien varias fotos (inscribir con anadir=True).")
        self.bd.anadir(id, [self._transf.aplicar(t) for t in ts], reemplazar=reemplazar)

    def verificar_t(self, id, template):
        t = validar_template(template)
        if not isinstance(id, str):
            return False, 0.0
        espera = self.intentos.bloqueado(id)
        if espera > 0:
            warnings.warn(f"Demasiados intentos fallidos con este id: bloqueado {espera:.0f} s mas")
            return False, 0.0
        ok, score = self._verificar_raw(id, t)
        self.intentos.registrar(id, ok)
        return ok, cuantizar(score if score is not None else 0.0, self.paso_score)

    def identificar_t(self, template):
        id_, score = self._identificar_raw(validar_template(template))
        return id_, cuantizar(score, self.paso_score)

    # ============================================================ con detector
    def _templates_fotos(self, foto, con_detector: bool):
        fotos = list(foto) if isinstance(foto, (list, tuple)) else [foto]
        return [self._template(f, con_detector) for f in fotos]

    def inscribir(self, id, foto, reemplazar: bool = False, **kw):
        """`foto` puede ser una foto o una lista de fotos de la misma persona."""
        self.inscribir_t(id, self._templates_fotos(foto, True), reemplazar, **kw)

    def verificar(self, id, foto):
        try:
            t = self._template(foto, con_detector=True)
        except CaraNoDetectada:
            return False, None
        return self.verificar_t(id, t)

    def identificar(self, foto):
        try:
            t = self._template(foto, con_detector=True)
        except CaraNoDetectada:
            return None, None
        return self.identificar_t(t)

    # ============================================================ sin detector
    def inscribir_sd(self, id, foto, reemplazar: bool = False, **kw):
        self.inscribir_t(id, self._templates_fotos(foto, False), reemplazar, **kw)

    def verificar_sd(self, id, foto):
        try:
            t = self._template(foto, con_detector=False)
        except CaraNoDetectada:
            return False, None
        return self.verificar_t(id, t)

    def identificar_sd(self, foto):
        try:
            t = self._template(foto, con_detector=False)
        except CaraNoDetectada:
            return None, None
        return self.identificar_t(t)
