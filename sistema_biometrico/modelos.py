"""Carga de los tres modelos del pipeline.

    foto --(1) YuNet (detector)--> recorte cuadrado de la cara
         --(2) CVLFace DFA MobileNet (alineador, 5 landmarks)--> cara alineada 112x112 + keypoints
         --(3) CVLFace ViT-Base KPRPE (AdaFace, WebFace4M)--> template de 512 floats (L2-normalizado)

Los modelos de CVLFace se distribuyen como "remote code" de HuggingFace. Ambos repos traen
paquetes Python de primer nivel (`models`, `aligners`) y rutas relativas (`pretrained_model/...`),
asi que hay que cargarlos con el cwd y el sys.path apuntando a su carpeta.
"""
from __future__ import annotations

import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
DIR_PESOS = Path(os.environ.get("SB_PESOS", RAIZ / "pretrained"))

REPO_KPRPE = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
REPO_DFA = "minchul/cvlface_DFA_mobilenet"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


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
    with _en_carpeta(path):
        modelo = AutoModel.from_pretrained(str(path), trust_remote_code=True)
    return modelo.eval()


class Detector:
    """Detector de caras YuNet (OpenCV). Devuelve la caja de la cara principal."""

    def __init__(self, ruta_onnx: Path | None = None, umbral_score: float = 0.6):
        import cv2
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
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
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
