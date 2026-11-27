"""Carga de los tres modelos del pipeline.

    foto --(1) YuNet (detector)--> recorte cuadrado de la cara
         --(2) CVLFace DFA MobileNet (alineador, 5 landmarks)--> cara alineada 112x112 + keypoints
         --(3) CVLFace ViT-Base KPRPE (AdaFace, WebFace12M)--> template de 512 floats (L2-normalizado)

Los modelos de CVLFace se distribuyen como "remote code" de HuggingFace. Ambos repos traen
paquetes Python de primer nivel (`models`, `aligners`) y rutas relativas (`pretrained_model/...`),
asi que hay que cargarlos con el cwd y el sys.path apuntando a su carpeta.
"""
from __future__ import annotations

import os
import sys
import contextlib
import io
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
DIR_PESOS = Path(os.environ.get("SB_PESOS", RAIZ / "pretrained"))

# Extractor: ViT-Base KPRPE de CVLFace. Dos versiones con la misma arquitectura, entrenadas con
# WebFace4M o WebFace12M (por defecto; SB_KPRPE=webface4m para la otra).
VARIANTES_KPRPE = {
    "webface4m": ("minchul/cvlface_adaface_vit_base_kprpe_webface4m",
                  "6530d73fb0af4d1d8287f31d559780c648ebd22a",
                  "3c6d37ea874c2f38ffc9a7f0e9247efc994c3fb5c12d044759ac294e19d127f7"),
    "webface12m": ("minchul/cvlface_adaface_vit_base_kprpe_webface12m",
                   "daefd5012d369588bd214fbaf4cc6b1d286e7066",
                   "99d16ed4aac0fdf0fcc82526b9b70703f3ec8c3041bf1bf44bd22751536e65db"),
}
VARIANTE = os.environ.get("SB_KPRPE", "webface12m")
if VARIANTE not in VARIANTES_KPRPE:
    raise ValueError(f"SB_KPRPE debe ser uno de {list(VARIANTES_KPRPE)}")
REPO_KPRPE = VARIANTES_KPRPE[VARIANTE][0]
REPO_DFA = "minchul/cvlface_DFA_mobilenet"
# Versiones fijadas (commit) y hash de los pesos: el build es reproducible y no se aceptan
# pesos cambiados aguas arriba o manipulados.
REVISION = {REPO_KPRPE: VARIANTES_KPRPE[VARIANTE][1],
            REPO_DFA: "8317e6dda53d91e7074979923144c2cc08906a33"}
SHA256 = {REPO_KPRPE: VARIANTES_KPRPE[VARIANTE][2],
          REPO_DFA: "80b6e922e4c76c10d5e24061fe47cd96112d18689bf5ae7e34af52e641c18c4a",
          "yunet": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"}
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")


@contextlib.contextmanager
def _en_carpeta(path: Path):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path.remove(str(path))


def _cargar_cvlface(path: Path):
    from transformers import AutoModel
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(
            f"No se encuentran los pesos en {path}. Ejecuta: python scripts/descargar_modelos.py")
    # El codigo remoto de CVLFace imprime mensajes de carga por stdout: se descartan para que la
    # salida de la CLI (p.ej. `main.py template`) sea solo el resultado. Los errores siguen saltando.
    with _en_carpeta(path), contextlib.redirect_stdout(io.StringIO()):
        modelo = AutoModel.from_pretrained(str(path), trust_remote_code=True)
    return modelo.eval()


class Detector:
    """Detector de caras YuNet (OpenCV). Devuelve la caja de la cara principal."""

    def __init__(self, ruta_onnx: Path | None = None, umbral_score: float = 0.6):
        import cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        self.cv2 = cv2
        ruta_onnx = Path(ruta_onnx or DIR_PESOS / "yunet" / "face_detection_yunet_2023mar.onnx")
        # Se carga desde buffer porque cv2 no abre rutas con caracteres no ASCII en Windows.
        buf = np.frombuffer(ruta_onnx.read_bytes(), dtype=np.uint8)
        self.net = cv2.FaceDetectorYN.create("onnx", buf, np.array([], dtype=np.uint8),
                                             (320, 320), umbral_score, 0.3, 5000)

    def detectar(self, rgb: np.ndarray) -> list[dict]:
        """Devuelve lista de caras {'bbox': (x, y, w, h), 'score': s, 'landmarks': (5,2)},
        con la cara principal en primer lugar."""
        h, w = rgb.shape[:2]
        # YuNet funciona mejor con imagenes de tamano moderado.
        escala = min(1.0, 1280 / max(h, w))
        img = rgb if escala == 1.0 else self.cv2.resize(rgb, (int(w * escala), int(h * escala)))
        bgr = img[:, :, ::-1].copy()
        self.net.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, caras = self.net.detect(bgr)
        if caras is None:
            return []
        res = []
        for c in caras:
            x, y, cw, ch = (c[:4] / escala).tolist()
            res.append({"bbox": (x, y, cw, ch), "score": float(c[14]),
                        "landmarks": (c[4:14].reshape(5, 2) / escala)})
        # Cara principal = la mas grande entre las de confianza alta (>= 0.8); asi una deteccion
        # espuria grande y dudosa no gana a la cara real. Si ninguna es de confianza alta, la mas grande.
        res.sort(key=lambda d: (d["score"] >= 0.8, d["bbox"][2] * d["bbox"][3]), reverse=True)
        return res


class Extractor:
    """Alineador DFA + ViT-KPRPE. Recibe un recorte de cara y devuelve el template."""

    def __init__(self, dispositivo: str | None = None):
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alineador = _cargar_cvlface(DIR_PESOS / REPO_DFA.split("/")[1]).to(self.dispositivo)
        self.modelo = _cargar_cvlface(DIR_PESOS / REPO_KPRPE.split("/")[1]).to(self.dispositivo)

    @staticmethod
    def a_tensor(rgb: np.ndarray) -> torch.Tensor:
        """uint8 HxWx3 RGB -> tensor 1x3xHxW normalizado a [-1, 1]."""
        t = torch.from_numpy(np.require(rgb, requirements=["C", "W"])).permute(2, 0, 1).float() / 255.0
        return ((t - 0.5) / 0.5).unsqueeze(0)

    @torch.no_grad()
    def alinear(self, rgb_cuadrado: np.ndarray):
        x = self.a_tensor(rgb_cuadrado).to(self.dispositivo)
        alineada, _, ldmks_alineados, score, _, _ = self.alineador(x)
        return alineada, ldmks_alineados, float(score.flatten()[0])

    @torch.no_grad()
    def extraer(self, rgb_cuadrado: np.ndarray, usar_flip: bool = True):
        """Devuelve (template L2-normalizado np.float32[512], score_alineador, cara_alineada_uint8)."""
        alineada, kps, score = self.alinear(rgb_cuadrado)
        emb = self.modelo(alineada, kps)
        if usar_flip:
            # Test-time augmentation: media con la imagen espejada (keypoints espejados).
            kps_f = kps.clone()
            kps_f[..., 0] = 1.0 - kps_f[..., 0]
            kps_f = kps_f[:, [1, 0, 2, 4, 3], :]  # intercambiar ojos y comisuras
            emb = emb + self.modelo(alineada.flip(3), kps_f)
        emb = torch.nn.functional.normalize(emb, dim=1)[0].cpu().numpy().astype(np.float32)
        cara = ((alineada[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255)
        return emb, score, cara.astype(np.uint8)
