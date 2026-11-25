"""Evaluacion del sistema propio: rendimiento, gemelas y ataques.

Genera resultados/informe.md, resultados/metricas.json y figuras en resultados/.

Datos que usa (los opcionales se aprovechan si existen):
  bd/base_datos.csv                 BD con las 51 personas (scripts/inscribir_dataset.py)
  data/Dataset, data/propia         fotos de inscripcion (para pares genuinos sinteticos)
  data/test/<id>/*.jpg              [opcional] otras fotos de personas inscritas -> pares genuinos REALES
  data/desconocidos/*.jpg           [opcional] personas NO inscritas (p.ej. Ashley Olsen, Oliver Phelps)

Uso:  python scripts/evaluar.py
"""
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import io

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sistema_biometrico import SistemaBiometrico, CaraNoDetectada  # noqa: E402
from sistema_biometrico.base_datos import BaseDatos  # noqa: E402
from inscribir_dataset import fotos_inscripcion, id_desde_fichero, EXT  # noqa: E402

OUT = RAIZ / "resultados"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
M = {}            # metricas -> json
INF = []          # lineas del informe


def h(t):
    INF.append(f"\n## {t}\n")
    print("\n==", t)


def p(t=""):
    INF.append(t)
    print(t)


# ------------------------------------------------------------- transformaciones
def _jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _ruido(img, s):
    a = np.asarray(img).astype(np.float32) + RNG.normal(0, s, (img.height, img.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _baja_res(img, lado):
    f = lado / max(img.size)
    return img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)


def _ocluir(img, sistema, zona):
    """Tapa ojos (gafas de sol) o mitad inferior (mascarilla) usando la deteccion."""
    caras = sistema.detector.detectar(np.asarray(img))
    if not caras:
        return img
    x, y, w, hh = caras[0]["bbox"]
    lm = caras[0]["landmarks"]
    im = img.copy()
    d = ImageDraw.Draw(im)
    if zona == "ojos":
        cy = (lm[0, 1] + lm[1, 1]) / 2
        d.rectangle([x, cy - 0.12 * hh, x + w, cy + 0.12 * hh], fill=(0, 0, 0))
    else:
        d.rectangle([x, lm[2, 1], x + w, y + hh * 1.05], fill=(90, 140, 200))
    return im


TRANSF = {
    "original":        lambda im, s: im,
    "espejo":          lambda im, s: ImageOps.mirror(im),
    "rotacion 15":     lambda im, s: im.rotate(15, resample=Image.BILINEAR, expand=True),
    "rotacion 30":     lambda im, s: im.rotate(30, resample=Image.BILINEAR, expand=True),
    "rotacion 90":     lambda im, s: im.rotate(90, expand=True),
    "baja res 200px":  lambda im, s: _baja_res(im, 200),
    "baja res 100px":  lambda im, s: _baja_res(im, 100),
    "desenfoque r=3":  lambda im, s: im.filter(ImageFilter.GaussianBlur(3)),
    "oscura x0.4":     lambda im, s: ImageEnhance.Brightness(im).enhance(0.4),
    "sobreexpuesta x1.8": lambda im, s: ImageEnhance.Brightness(im).enhance(1.8),
    "escala de grises": lambda im, s: ImageOps.grayscale(im).convert("RGB"),
    "JPEG q=10":       lambda im, s: _jpeg(im, 10),
    "ruido sigma=25":  lambda im, s: _ruido(im, 25),
    "gafas de sol":    lambda im, s: _ocluir(im, s, "ojos"),
    "mascarilla":      lambda im, s: _ocluir(im, s, "boca"),
}


def far_frr(gen, imp, umbrales):
    gen, imp = np.asarray(gen), np.asarray(imp)
    far = np.array([(imp >= u).mean() for u in umbrales])
    frr = np.array([(gen < u).mean() for u in umbrales])
    return far, frr


def eer(gen, imp):
    us = np.linspace(-1, 1, 2001)
    far, frr = far_frr(gen, imp, us)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(us[i])


def main():
    bd_orig = RAIZ / "bd" / "base_datos.csv"
    if not bd_orig.exists():
        sys.exit("Primero: python scripts/inscribir_dataset.py")
    bd_eval = OUT / "bd_eval.csv"
    shutil.copy(bd_orig, bd_eval)
    os.environ.pop("SB_CLAVE_HMAC", None)
    s = SistemaBiometrico(bd_eval)
    U = s.umbral_verificacion
    ids = s.bd.ids
    T = s.bd.templates
    INF.append("# Informe de evaluacion - CVLFace ViT-KPRPE\n")
    p(f"Personas inscritas: **{len(set(ids))}** | umbral verificacion = **{U}** | "
      f"umbral identificacion = **{s.umbral_identificacion}** (similitud coseno)")

    # ================================================================ impostores
    h("1. Distribucion de impostores (inscritos entre si)")
    S = T @ T.T
    iu = np.triu_indices(len(ids), 1)
    imp = S[iu]
    M["impostores"] = {"n": int(len(imp)), "media": float(imp.mean()), "max": float(imp.max()),
                       "p99": float(np.percentile(imp, 99)), "FAR@umbral": float((imp >= U).mean())}
    p(f"{len(imp)} pares. media={imp.mean():.3f}, p99={np.percentile(imp, 99):.3f}, "
      f"max={imp.max():.3f}, FAR al umbral = {(imp >= U).mean():.4f}")
    p("\nPares de impostores mas parecidos:\n\n| persona A | persona B | score |\n|---|---|---|")
    for k in np.argsort(-imp)[:8]:
        p(f"| {ids[iu[0][k]]} | {ids[iu[1][k]]} | {imp[k]:.3f} |")

    # ================================================================ robustez
    h("2. Pares genuinos sinteticos (foto de inscripcion transformada)")
    p("Cada foto de inscripcion se transforma y se verifica contra su propio template. Es una "
      "estimacion OPTIMISTA (misma foto de origen), pero mide la robustez a cada degradacion.\n")
    fotos = list(fotos_inscripcion())
    rob, gen_sint = {}, []
    for nombre, f_t in TRANSF.items():
        sc, fallos_det, acier_id = [], 0, 0
        for f in fotos:
            id_ = id_desde_fichero(f)
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            try:
                t = s.extraer_template(f_t(im, s))
            except CaraNoDetectada:
                fallos_det += 1
                sc.append(-1.0)
                continue
            sc.append(float(np.max(s.bd.templates_de(id_) @ t)))
            acier_id += int(s.identificar_t(t)[0] == id_)
        sc = np.array(sc)
        if nombre != "original":
            gen_sint += sc.tolist()
        rob[nombre] = {"score_medio": float(sc.mean()), "score_min": float(sc.min()),
                       "TAR@umbral": float((sc >= U).mean()), "sin_cara": fallos_det,
                       "rank1_identificacion": acier_id / len(fotos)}
        print(f"  {nombre:20s} {rob[nombre]}")
    M["robustez"] = rob
    p("| transformacion | score medio | score min | TAR (verif.) | identif. correcta | sin cara detectada |")
    p("|---|---|---|---|---|---|")
    for n, r in rob.items():
        p(f"| {n} | {r['score_medio']:.3f} | {r['score_min']:.3f} | {r['TAR@umbral']:.0%} | "
          f"{r['rank1_identificacion']:.0%} | {r['sin_cara']} |")

    # ================================================================ genuinos reales
    gen_real, imp_real, ident_real = [], [], []
    dir_test = RAIZ / "data" / "test"
    if dir_test.exists():
        h("3. Pares genuinos REALES (data/test)")
        for carpeta in sorted(d for d in dir_test.iterdir() if d.is_dir()):
            id_ = carpeta.name
            if id_ not in s.bd:
                p(f"- aviso: '{id_}' no esta inscrito, se ignora")
                continue
            for f in sorted(x for x in carpeta.iterdir() if x.suffix.lower() in EXT):
                try:
                    t = s.extraer_template(f)
                except CaraNoDetectada:
                    gen_real.append(-1.0)
                    ident_real.append((id_, None, f.name))
                    continue
                sc_all = T @ t
                gen_real.append(float(np.max(s.bd.templates_de(id_) @ t)))
                imp_real += [float(v) for v, x in zip(sc_all, ids) if x != id_]
                ident_real.append((id_, s.identificar_t(t)[0], f.name))
        if gen_real:
            gr, ir = np.array(gen_real), np.array(imp_real)
            e, ue = eer(gr, ir)
            M["real"] = {"genuinos": len(gr), "impostores": len(ir), "TAR@umbral": float((gr >= U).mean()),
                         "FAR@umbral": float((ir >= U).mean()), "EER": e, "umbral_EER": ue,
                         "rank1": float(np.mean([a == b for a, b, _ in ident_real]))}
            p(f"{len(gr)} genuinos, {len(ir)} impostores. TAR={M['real']['TAR@umbral']:.1%}, "
              f"FAR={M['real']['FAR@umbral']:.2%} al umbral {U}. EER={e:.2%} (umbral {ue:.2f}). "
              f"Identificacion correcta (open-set) = {M['real']['rank1']:.1%}")
            fallos = [x for x in ident_real if x[0] != x[1]]
            if fallos:
                p("\nErrores de identificacion:\n\n| real | identificado | foto |\n|---|---|---|")
                for a, b, n in fallos:
                    p(f"| {a} | {b or 'DESCONOCIDO'} | {n} |")
    else:
        p("\n> No hay data/test/: no se calculan pares genuinos reales (ver README).")

    # ================================================================ desconocidos
    dir_desc = RAIZ / "data" / "desconocidos"
    if dir_desc.exists():
        h("4. Personas NO inscritas (identificacion en conjunto abierto)")
        p("| foto | identificado como | score top-1 | top-2 |\n|---|---|---|---|")
        fa = 0
        res = []
        for f in sorted(x for x in dir_desc.iterdir() if x.suffix.lower() in EXT):
            try:
                t = s.extraer_template(f)
            except CaraNoDetectada:
                p(f"| {f.name} | (sin cara) | - | - |")
                continue
            r = s.ranking_t(t, 2)
            id_, _ = s.identificar_t(t)
            fa += id_ is not None
            res.append({"foto": f.name, "identificado": id_, "top": r})
            p(f"| {f.name} | **{id_ or 'DESCONOCIDO'}** | {r[0][0]} {r[0][1]:.3f} | {r[1][0]} {r[1][1]:.3f} |")
        M["desconocidos"] = {"n": len(res), "falsas_aceptaciones": fa, "detalle": res}

    # ================================================================ gemelas
    h("5. Gemelas Twin Melody (Paula vs Aitana)")
    id_p = next((i for i in ids if "paula" in i.lower()), None)
    id_a = next((i for i in ids if "aitana" in i.lower()), None)
    if id_p and id_a:
        sc = float(np.max(s.bd.templates_de(id_p) @ s.bd.templates_de(id_a).T))
        rank = int((imp > sc).sum())
        M["gemelas"] = {"score": sc, "supera_umbral": sc >= U, "pares_impostores_mas_altos": rank}
        p(f"Similitud entre los templates de **{id_p}** y **{id_a}**: **{sc:.3f}** "
          f"({'SUPERA' if sc >= U else 'no supera'} el umbral {U}). "
          f"Hay {rank} pares de impostores con score mayor.")
        filas = []
        for real, otra in ((id_p, id_a), (id_a, id_p)):
            d = RAIZ / "data" / "test" / real
            for f in (sorted(x for x in d.iterdir() if x.suffix.lower() in EXT) if d.exists() else []):
                t = s.extraer_template(f)
                sp_ = float(np.max(s.bd.templates_de(real) @ t))
                so = float(np.max(s.bd.templates_de(otra) @ t))
                filas.append({"real": real, "foto": f.name, "score_propia": sp_, "score_hermana": so,
                              "identificada_como": s.identificar_t(t)[0]})
        if filas:
            p("\nFotos de prueba de cada gemela verificadas contra su id y contra el de su hermana:\n")
            p("| persona real | foto | score con su id | score con la hermana | aceptada como hermana | identificada como |")
            p("|---|---|---|---|---|---|")
            for r in filas:
                p(f"| {r['real']} | {r['foto']} | {r['score_propia']:.3f} | {r['score_hermana']:.3f} | "
                  f"{'SI' if r['score_hermana'] >= U else 'no'} | {r['identificada_como']} |")
            fa = np.mean([r["score_hermana"] >= U for r in filas])
            p(f"\nFAR entre gemelas = **{fa:.0%}**: la gemela entra como su hermana en verificacion.")
            M["gemelas"]["pruebas"] = filas
            M["gemelas"]["FAR_gemelas"] = float(fa)
    else:
        p("> Falta la foto de Paula Etxeberria en data/propia/.")

    # ================================================================ ataques
    h("6. Ataques")
    A = {}

    # 6.1 saltarse el detector con entradas que no son caras
    p("\n### 6.1 Saltarse el detector (*_sd) con imagenes que no son caras\n")
    p("| entrada | con detector | sin detector: id top-1 | score | aceptado |\n|---|---|---|---|---|")
    basura = {
        "ruido uniforme": Image.fromarray(RNG.integers(0, 256, (256, 256, 3), dtype=np.uint8)),
        "gris liso": Image.new("RGB", (256, 256), (128, 128, 128)),
        "negro": Image.new("RGB", (256, 256)),
        "degradado": Image.fromarray(np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None], (256, 1, 3))),
    }
    A["sd_no_caras"] = {}
    for n, im in basura.items():
        id_d, _ = s.identificar(im)
        id_, sc = s.identificar_sd(im)
        A["sd_no_caras"][n] = {"con_detector": id_d, "sd_id": id_, "sd_score": sc}
        p(f"| {n} | {'rechazado (sin cara)' if id_d is None else id_d} | {s.ranking_t(s.extraer_template(im, False), 1)[0][0]} | {sc:.3f} | {id_ is not None} |")

    # 6.1b: fotos completas por *_sd (sin recortar)
    ok = 0
    for f in fotos:
        id_, _ = s.identificar_sd(f)
        ok += id_ == id_desde_fichero(f)
    A["sd_fotos_completas_rank1"] = ok / len(fotos)
    p(f"\nFotos de inscripcion completas (sin recortar) por identificar_sd: "
      f"{ok}/{len(fotos)} identificadas correctamente. El alineador DFA asume una cara centrada que "
      f"ocupa la imagen; si la cara es pequena el resultado se degrada.")

    # 6.2 lectura de la BD -> replay de templates
    p("\n### 6.2 El atacante LEE la base de datos (replay de templates)\n")
    acept = sum(s.verificar_t(i, t)[0] for i, t in zip(ids, T))
    ident = sum(s.identificar_t(t)[0] == i for i, t in zip(ids, T))
    A["replay"] = {"verificaciones_aceptadas": acept, "identificaciones": ident, "n": len(ids)}
    p(f"Reinyectando los templates robados con verificar_t/identificar_t: {acept}/{len(ids)} "
      f"verificaciones aceptadas, {ident}/{len(ids)} identificaciones (score 1.0). Los templates "
      f"en claro equivalen a credenciales: con verificar_t no hace falta ni la cara.")

    # 6.3 hill-climbing con el score como oraculo
    p("\n### 6.3 Ataque de hill-climbing usando el score de verificar_t como oraculo\n")
    p("El atacante NO lee la BD; solo consulta verificar_t(victima, template) y ve el score.\n")
    hc = []
    for victima in list(dict.fromkeys(ids))[:5]:
        t = RNG.normal(size=512)
        _, best = s.verificar_t(victima, t)
        q = 1
        while best < U and q < 20000:
            cand = t + RNG.normal(scale=0.3, size=512)
            _, sc = s.verificar_t(victima, cand)
            q += 1
            if sc > best:
                t, best = cand, sc
        hc.append({"victima": victima, "consultas": q, "score_final": best, "aceptado": best >= U})
        p(f"- {victima}: aceptado={best >= U} tras {q} consultas (score {best:.3f})")
    A["hill_climbing"] = hc

    # 6.4 plantilla "maestra"
    p("\n### 6.4 Template 'maestro' (media de todos los templates de la BD)\n")
    media = T.mean(0)
    sc_m = T @ (media / np.linalg.norm(media))
    A["maestro"] = {"aceptado_por": int((sc_m >= U).sum()), "score_max": float(sc_m.max())}
    p(f"Score maximo contra la BD = {sc_m.max():.3f}; aceptado por {int((sc_m >= U).sum())} identidades.")

    # 6.5 escritura en la BD
    p("\n### 6.5 El atacante ESCRIBE en la base de datos\n")
    victima = ids[0]
    atacante = T[1]  # el atacante usa su propia cara (aqui: otra persona inscrita)
    p(f"Victima: {victima}. El atacante anade una fila con SU template bajo el id de la victima.")
    for con_clave in (False, True):
        bd_tmp = OUT / f"bd_ataque_{int(con_clave)}.csv"
        if con_clave:
            os.environ["SB_CLAVE_HMAC"] = "clave-secreta-del-sistema"
        BaseDatos(bd_tmp).ruta.unlink(missing_ok=True)
        b = BaseDatos(bd_tmp)
        b.ids, b.templates = list(ids), T.copy()
        b.guardar()  # BD legitima (firmada si hay clave)
        with open(bd_tmp, "a", encoding="utf-8", newline="") as fh:  # escritura del atacante
            fh.write(f"{victima},\"{';'.join(f'{v:.8f}' for v in atacante)}\",\n")
        s2 = SistemaBiometrico.__new__(SistemaBiometrico)
        s2.bd, s2.umbral_verificacion, s2.umbral_identificacion = BaseDatos(bd_tmp), U, s.umbral_identificacion
        ok, sc = s2.verificar_t(victima, atacante)
        A[f"escritura_hmac_{con_clave}"] = {"atacante_aceptado": ok, "filas_rechazadas": s2.bd.filas_rechazadas}
        p(f"- {'CON' if con_clave else 'SIN'} firma HMAC: atacante aceptado como victima = **{ok}** "
          f"(score {sc:.3f}); filas descartadas al cargar = {s2.bd.filas_rechazadas}")
        os.environ.pop("SB_CLAVE_HMAC", None)
        bd_tmp.unlink(missing_ok=True)
    p("La firma HMAC solo protege si la clave NO esta al alcance del atacante (p.ej. variable de "
      "entorno del servidor, no en el repositorio). Si puede borrar filas sigue siendo posible una "
      "denegacion de servicio, y si conoce la clave el ataque funciona igual.")

    # 6.6 validacion de entradas
    p("\n### 6.6 Templates malformados\n")
    for n, t in {"NaN": np.full(512, np.nan), "longitud 128": np.ones(128), "vector nulo": np.zeros(512),
                 "texto": "hola"}.items():
        try:
            s.verificar_t(ids[0], t)
            r = "ACEPTADO (fallo)"
        except Exception as e:
            r = f"rechazado: {type(e).__name__}: {e}"
        p(f"- {n}: {r}")

    # 6.7 enumeracion de ids
    ok_desc, sc_desc = s.verificar_t("persona_que_no_existe", T[0])
    p(f"\n### 6.7 Enumeracion de identidades\n\nverificar_t con un id inexistente devuelve score={sc_desc} "
      f"(None) mientras que con un id existente devuelve un numero: se puede averiguar quien esta inscrito.")
    M["ataques"] = A

    # ================================================================ figura
    plt.figure(figsize=(8, 4))
    plt.hist(imp, bins=60, alpha=.6, density=True, label=f"impostores inscritos (n={len(imp)})")
    plt.hist(gen_sint, bins=60, alpha=.6, density=True, label="genuinos sinteticos (transformaciones)")
    if gen_real:
        plt.hist(gen_real, bins=30, alpha=.6, density=True, label=f"genuinos reales (n={len(gen_real)})")
    if "gemelas" in M:
        plt.axvline(M["gemelas"]["score"], color="m", ls=":", label="Paula vs Aitana")
    plt.axvline(U, color="k", ls="--", label=f"umbral {U}")
    plt.xlabel("similitud coseno")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "distribuciones.png", dpi=120)
    INF.append("\n![distribuciones](distribuciones.png)\n")

    (OUT / "informe.md").write_text("\n".join(INF), encoding="utf-8")
    (OUT / "metricas.json").write_text(json.dumps(M, indent=2, default=str), encoding="utf-8")
    bd_eval.unlink(missing_ok=True)
    print("\nInforme en", OUT / "informe.md")


if __name__ == "__main__":
    main()
