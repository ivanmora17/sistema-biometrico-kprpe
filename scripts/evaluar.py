"""Evaluacion del sistema propio: rendimiento, gemelas y ataques.

Genera resultados/informe.md, resultados/metricas.json y figuras en resultados/.

Datos que usa (los opcionales se aprovechan si existen):
  bd/base_datos.csv                 BD con las 50 personas (scripts/inscribir_dataset.py)
  data/Dataset, data/propia         fotos de inscripcion (para pares genuinos sinteticos)
  data/test/<id>/*.jpg              [opcional] otras fotos de personas inscritas -> pares genuinos REALES
  data/extra/<id>/*.jpg             templates adicionales (gemelas); se evaluan con leave-one-out
  data/desconocidos/*.jpg           [opcional] personas NO inscritas (p.ej. Ashley Olsen, Oliver Phelps)
  data/descargas/*.jpg              fotos de las gemelas JUNTAS (ataque de varias caras)

Las metricas se calculan con los scores SIN cuantizar (metodos internos _scores/_verificar_raw)
y sin bloqueo por intentos; los ataques usan la API publica con las defensas activas.

Uso:  python scripts/evaluar.py
"""
import base64
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

from sistema_biometrico import SistemaBiometrico, CaraNoDetectada, ErrorIntegridad  # noqa: E402
from sistema_biometrico.sistema import LADO_MIN_CARA  # noqa: E402
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



def copiar_bd(origen: Path, destino: Path):
    for suf in ("", ".sello"):
        shutil.copy(str(origen) + suf, str(destino) + suf)
    destino.with_name(destino.stem + ".intentos.json").unlink(missing_ok=True)


def borrar_bd(ruta: Path):
    for suf in ("", ".sello"):
        Path(str(ruta) + suf).unlink(missing_ok=True)
    ruta.with_name(ruta.stem + ".intentos.json").unlink(missing_ok=True)


def main():
    bd_orig = RAIZ / "bd" / "base_datos.csv"
    if not bd_orig.exists():
        sys.exit("Primero: python scripts/inscribir_dataset.py")
    bd_eval = OUT / "bd_eval.csv"
    copiar_bd(bd_orig, bd_eval)
    s = SistemaBiometrico(bd_eval, max_intentos=0)  # metricas: sin bloqueo
    U = s.umbral_verificacion
    ids = s.bd.ids
    T = s.bd.templates  # templates PROTEGIDOS (R·t); el producto escalar se conserva

    def sc_id(t, id_):
        return s._por_identidad(s._scores(t))[id_]

    INF.append("# Informe de evaluacion - CVLFace ViT-KPRPE\n")
    p(f"Personas inscritas: **{len(set(ids))}** | umbral verificacion = **{U}** | "
      f"umbral identificacion = **{s.umbral_identificacion}** (similitud coseno) | "
      f"margen de ambiguedad = **{s.margen}**")
    p("\nLas metricas usan el score real (sin cuantizar) y la regla de decision completa: umbral + "
      "rechazo si otra identidad inscrita queda a menos del margen. Ver seccion 6 para las defensas.")

    # ================================================================ impostores
    h("1. Distribucion de impostores (inscritos entre si)")
    # Un template por identidad (el de la foto de inscripcion principal); las gemelas tienen ademas
    # templates extra (data/extra) que no deben contar como "impostores" entre si.
    prim = [ids.index(i) for i in dict.fromkeys(ids)]
    ids_p, T_p = [ids[k] for k in prim], T[prim]
    S = T_p @ T_p.T
    iu = np.triu_indices(len(ids_p), 1)
    imp = S[iu]
    M["impostores"] = {"n": int(len(imp)), "media": float(imp.mean()), "max": float(imp.max()),
                       "p99": float(np.percentile(imp, 99)), "FAR@umbral": float((imp >= U).mean())}
    p(f"{len(imp)} pares. media={imp.mean():.3f}, p99={np.percentile(imp, 99):.3f}, "
      f"max={imp.max():.3f}, pares por encima del umbral = {(imp >= U).mean():.4f}")
    p("\nPares de impostores mas parecidos:\n\n| persona A | persona B | score |\n|---|---|---|")
    for k in np.argsort(-imp)[:8]:
        p(f"| {ids_p[iu[0][k]]} | {ids_p[iu[1][k]]} | {imp[k]:.3f} |")

    # ================================================================ robustez
    h("2. Pares genuinos sinteticos (foto de inscripcion transformada)")
    p("Cada foto de inscripcion se transforma y se verifica contra su propio id. Es una estimacion "
      "OPTIMISTA (misma foto de origen), pero mide la robustez a cada degradacion. Las fotos que el "
      f"control de calidad rechaza (sin cara, varias caras, cara < {LADO_MIN_CARA} px, mala alineacion) cuentan "
      "como rechazo en TAR, pero no entran en las columnas de score.\n")
    fotos = list(fotos_inscripcion())
    rob, gen_sint = {}, []
    for nombre, f_t in TRANSF.items():
        sc, sin_cara, acept, acier_id = [], 0, 0, 0
        for f in fotos:
            id_ = id_desde_fichero(f)
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            try:
                t = s.extraer_template(f_t(im, s))
            except CaraNoDetectada:
                sin_cara += 1
                continue
            sc.append(sc_id(t, id_))
            acept += int(s._verificar_raw(id_, t)[0])
            acier_id += int(s._identificar_raw(t)[0] == id_)
        sc = np.array(sc) if sc else np.array([np.nan])
        if nombre != "original":
            gen_sint += [v for v in sc.tolist() if np.isfinite(v)]
        rob[nombre] = {"score_medio": float(np.nanmean(sc)), "score_min": float(np.nanmin(sc)),
                       "TAR@umbral": acept / len(fotos), "sin_cara_valida": sin_cara,
                       "rank1_identificacion": acier_id / len(fotos)}
        print(f"  {nombre:20s} {rob[nombre]}")
    M["robustez"] = rob
    p("| transformacion | score medio | score min | TAR (verif.) | identif. correcta | sin cara valida |")
    p("|---|---|---|---|---|---|")
    for n, r in rob.items():
        p(f"| {n} | {r['score_medio']:.3f} | {r['score_min']:.3f} | {r['TAR@umbral']:.0%} | "
          f"{r['rank1_identificacion']:.0%} | {r['sin_cara_valida']} |")

    # ================================================================ genuinos reales
    gen_real, imp_real, ident_real, acept_real = [], [], [], []
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
                    acept_real.append(False)
                    ident_real.append((id_, None, f.name))
                    continue
                por_id = s._por_identidad(s._scores(t))
                gen_real.append(por_id[id_])
                imp_real += [v for x, v in por_id.items() if x != id_]
                acept_real.append(s._verificar_raw(id_, t)[0])
                ident_real.append((id_, s._identificar_raw(t)[0], f.name))
        if gen_real:
            gr, ir = np.array(gen_real), np.array(imp_real)
            e, ue = eer(gr, ir)
            M["real"] = {"genuinos": len(acept_real), "impostores": len(ir),
                         "TAR_solo_umbral": float((gr >= U).mean()),
                         "TAR_con_margen": float(np.mean(acept_real)),
                         "FAR_solo_umbral": float((ir >= U).mean()), "EER": e, "umbral_EER": ue,
                         "rank1": float(np.mean([a == b for a, b, _ in ident_real]))}
            p(f"{len(acept_real)} genuinos, {len(ir)} impostores. Solo con el umbral: TAR="
              f"{M['real']['TAR_solo_umbral']:.1%}, FAR={M['real']['FAR_solo_umbral']:.2%}, EER={e:.2%} "
              f"(umbral {ue:.2f}). Con la regla de margen: TAR={M['real']['TAR_con_margen']:.1%}. "
              f"Identificacion correcta (open-set) = {M['real']['rank1']:.1%}")
            p("\n> Solo hay fotos de prueba de las gemelas, asi que estas cifras miden sobre todo la "
              "confusion entre ellas. Para medir el FRR real hacen falta mas fotos en data/test/<id>/.")
            fallos = [x for x in ident_real if x[0] != x[1]]
            if fallos:
                p("\nErrores de identificacion (None = rechazada por ambigua):\n\n| real | identificado | foto |\n|---|---|---|")
                for a, b, n in fallos:
                    p(f"| {a} | {b or 'DESCONOCIDO / AMBIGUA'} | {n} |")
    else:
        p("\n> No hay data/test/: no se calculan pares genuinos reales (ver README).")

    # ================================================================ desconocidos
    dir_desc = RAIZ / "data" / "desconocidos"
    h("4. Personas NO inscritas (identificacion en conjunto abierto)")
    if dir_desc.exists():
        p("| foto | identificado como | top-1 | top-2 |\n|---|---|---|---|")
        fa, res = 0, []
        for f in sorted(x for x in dir_desc.iterdir() if x.suffix.lower() in EXT):
            try:
                t = s.extraer_template(f)
            except CaraNoDetectada as e:
                p(f"| {f.name} | (sin cara valida: {e}) | - | - |")
                continue
            r = s._ranking(t, 2)
            id_, _ = s._identificar_raw(t)
            fa += id_ is not None
            res.append({"foto": f.name, "identificado": id_, "top": r})
            p(f"| {f.name} | **{id_ or 'DESCONOCIDO'}** | {r[0][0]} {r[0][1]:.3f} | {r[1][0]} {r[1][1]:.3f} |")
        M["desconocidos"] = {"n": len(res), "falsas_aceptaciones": fa, "detalle": res}
    else:
        p("> No hay data/desconocidos/: no se ha probado con personas no inscritas.")

    # ================================================================ gemelas
    h("5. Gemelas Twin Melody (Paula vs Aitana)")
    id_p = next((i for i in ids if "paula" in i.lower()), None)
    id_a = next((i for i in ids if "aitana" in i.lower()), None)
    if id_p and id_a:
        ip, ia = ids.index(id_p), ids.index(id_a)
        sc = float(T[ip] @ T[ia])
        rank = int((imp > sc + 1e-5).sum())
        M["gemelas"] = {"score": sc, "supera_umbral": sc >= U, "pares_impostores_mas_altos": rank}
        p(f"Similitud entre las fotos de inscripcion de **{id_p}** y **{id_a}**: **{sc:.3f}** "
          f"({'SUPERA' if sc >= U else 'no supera'} el umbral {U}). "
          f"Hay {rank} pares de impostores con score mayor.")
        n_tp = sum(1 for i in ids if i == id_p)
        n_ta = sum(1 for i in ids if i == id_a)
        p(f"\nCada gemela esta inscrita con varias fotos (Paula: {n_tp}, Aitana: {n_ta}; ver data/extra). "
          f"Prueba **leave-one-out**: cada foto etiquetada de una gemela se compara con la BD SIN su propio "
          f"template, como si fuera una foto nueva. 'Antes' = sistema original (una foto por persona y solo "
          f"umbral); 'ahora' = varias fotos + margen de ambiguedad {s.margen}.\n")

        def fotos_de(i):
            base = [f for f in fotos if id_desde_fichero(f) == i]
            d = RAIZ / "data" / "extra" / i
            return base + (sorted(x for x in d.iterdir() if x.suffix.lower() in EXT) if d.exists() else [])

        def decidir(por_id, reclamada):
            sc_ = por_id.get(reclamada, -1.0)
            otra_ = max((v for k, v in por_id.items() if k != reclamada), default=-1.0)
            return sc_ >= U and not (otra_ >= U and sc_ - otra_ < s.margen)

        filas = []
        for real, otra in ((id_p, id_a), (id_a, id_p)):
            for f in fotos_de(real):
                t = s.extraer_template(f)
                scores = s._scores(t)
                mask = scores < 0.999  # quitar el template de esta misma foto
                por_id = {}
                for i, v in zip(np.array(ids)[mask], scores[mask].tolist()):
                    por_id[i] = max(v, por_id.get(i, -1.0))
                orden = sorted(por_id.items(), key=lambda kv: -kv[1])
                ident = orden[0][0] if decidir(por_id, orden[0][0]) else None
                antes_hermana = float(T[ids.index(otra)] @ s._transf.aplicar(t)) >= U
                filas.append({"real": real, "foto": f.name, "score_propia": por_id.get(real, -1.0),
                              "score_hermana": por_id[otra], "aceptada_propia": decidir(por_id, real),
                              "aceptada_hermana_antes": antes_hermana,
                              "aceptada_hermana": decidir(por_id, otra), "identificada_como": ident})
        p("| persona real | foto | score su id | score hermana | aceptada como ella | "
          "aceptada como hermana (antes) | aceptada como hermana (ahora) | identificada como |")
        p("|---|---|---|---|---|---|---|---|")
        for r in filas:
            p(f"| {r['real']} | {r['foto']} | {r['score_propia']:.3f} | {r['score_hermana']:.3f} | "
              f"{'SI' if r['aceptada_propia'] else 'no'} | {'SI' if r['aceptada_hermana_antes'] else 'no'} | "
              f"{'SI' if r['aceptada_hermana'] else 'no'} | {r['identificada_como'] or 'AMBIGUA'} |")
        M["gemelas"]["pruebas"] = filas
        M["gemelas"]["FAR_gemelas_antes"] = float(np.mean([r["aceptada_hermana_antes"] for r in filas]))
        M["gemelas"]["FAR_gemelas"] = float(np.mean([r["aceptada_hermana"] for r in filas]))
        M["gemelas"]["FRR_gemelas"] = float(1 - np.mean([r["aceptada_propia"] for r in filas]))
        M["gemelas"]["identificacion_correcta"] = float(np.mean([r["identificada_como"] == r["real"] for r in filas]))
        p(f"\nFAR entre gemelas: **{M['gemelas']['FAR_gemelas_antes']:.0%} antes -> "
          f"{M['gemelas']['FAR_gemelas']:.0%} ahora**. Aceptada como ella misma: "
          f"**{1 - M['gemelas']['FRR_gemelas']:.0%}**; identificacion correcta: "
          f"{M['gemelas']['identificacion_correcta']:.0%} (si la foto es ambigua se rechaza a las dos). "
          f"Solo hay {len(filas)} fotos etiquetadas: con mas fotos de cada gemela el sistema mejoraria.")
    else:
        p("> Falta la foto de Paula Etxeberria en data/propia/.")

    # ================================================================ ataques
    h("6. Ataques y defensas")
    A = {}
    # Sistema "de produccion" con todas las defensas (bloqueo incluido) sobre una copia de la BD.
    bd_def = OUT / "bd_defensas.csv"

    def sistema_defensas(**kw):
        copiar_bd(bd_orig, bd_def)
        s2 = SistemaBiometrico.__new__(SistemaBiometrico)
        s2.__dict__.update(s.__dict__)
        from sistema_biometrico.base_datos import BaseDatos
        from sistema_biometrico.seguridad import RegistroIntentos
        s2.bd = BaseDatos(bd_def, s._kh, s._kc)
        s2.intentos = RegistroIntentos(bd_def.with_name(bd_def.stem + ".intentos.json"), s._kh,
                                       kw.get("max_intentos", 5), kw.get("bloqueo_seg", 300))
        return s2

    # 6.1 saltarse el detector con entradas que no son caras
    p("\n### 6.1 Saltarse el detector (*_sd) con imagenes que no son caras\n")
    p("| entrada | con detector | sin detector (_sd) |\n|---|---|---|")
    basura = {
        "ruido uniforme": Image.fromarray(RNG.integers(0, 256, (256, 256, 3), dtype=np.uint8)),
        "gris liso": Image.new("RGB", (256, 256), (128, 128, 128)),
        "negro": Image.new("RGB", (256, 256)),
        "degradado": Image.fromarray(np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None], (256, 1, 3))),
    }
    A["sd_no_caras"] = {}
    for n, im in basura.items():
        r_d = s.identificar(im)
        r_sd = s.identificar_sd(im)
        A["sd_no_caras"][n] = {"con_detector": r_d, "sd": r_sd}
        fmt = lambda r: "rechazada (sin cara valida)" if r[1] is None else f"{r[0] or 'DESCONOCIDO'} ({r[1]})"
        p(f"| {n} | {fmt(r_d)} | {fmt(r_sd)} |")

    ok = inval = 0
    for f in fotos:
        id_, sc_ = s.identificar_sd(f)
        ok += id_ == id_desde_fichero(f)
        inval += sc_ is None
    A["sd_fotos_completas"] = {"correctas": ok, "rechazadas_calidad": inval, "n": len(fotos)}
    p(f"\nFotos de inscripcion completas (sin recortar) por identificar_sd: {ok}/{len(fotos)} bien "
      f"identificadas, {inval} rechazadas por el control de alineacion, {len(fotos) - ok - inval} "
      f"identificadas mal o como desconocidas. Antes se aceptaban todas y salian 24/50 bien.")

    # 6.2 lectura de la BD -> replay de templates
    p("\n### 6.2 El atacante LEE la base de datos (replay de templates)\n")
    p("El CSV va cifrado (AES-256-GCM): sin la clave no contiene ningun vector. Aqui se supone el "
      "peor caso: el atacante consigue los templates guardados ya descifrados (p.ej. leyendo la memoria "
      "del proceso) y los reinyecta.\n")
    acept = sum(s.verificar_t(i, t)[0] for i, t in zip(ids, T))
    ident = sum(s.identificar_t(t)[0] == i for i, t in zip(ids, T))
    A["replay"] = {"verificaciones_aceptadas": acept, "identificaciones": ident, "n": len(ids)}
    p(f"Reinyectando los templates guardados en verificar_t/identificar_t: {acept}/{len(ids)} "
      f"verificaciones aceptadas, {ident}/{len(ids)} identificaciones (antes: 50/50). La BD guarda "
      f"R·t con R secreta; verificar_t aplica R otra vez y R·R·t no se parece a R·t. "
      f"Quien tenga ademas la clave puede deshacer R (ver README).")

    # 6.3 hill-climbing
    p("\n### 6.3 Hill-climbing usando verificar_t como oraculo\n")
    p("El atacante NO lee la BD; prueba templates aleatorios y se queda con el que mejora el score "
      "(o lo iguala, para poder avanzar sobre las mesetas del score cuantizado).\n")

    def hill(oraculo, victima, presupuesto):
        t = RNG.normal(size=512)
        _, best = oraculo(victima, t)
        best = best or 0.0
        q = 1
        while q < presupuesto:
            cand = t + RNG.normal(scale=0.3, size=512)
            ok_, sc_ = oraculo(victima, cand)
            q += 1
            if ok_:
                return True, q
            if sc_ is not None and sc_ >= best:
                t, best = cand, sc_
        return False, q

    victimas = list(dict.fromkeys(ids))[:5]
    hc = {"score_exacto": [], "score_cuantizado": [], "con_bloqueo": []}
    s_cuant = sistema_defensas(max_intentos=0)
    s_bloq = sistema_defensas(max_intentos=5)
    p("| victima | score exacto (antes) | score cuantizado | cuantizado + bloqueo |\n|---|---|---|---|")
    for v in victimas:
        r1 = hill(lambda i, t: s._verificar_raw(i, t), v, 20000)
        r2 = hill(s_cuant.verificar_t, v, 5000)
        r3 = hill(s_bloq.verificar_t, v, 200)
        for k, r in zip(hc, (r1, r2, r3)):
            hc[k].append({"victima": v, "aceptado": r[0], "consultas": r[1]})
        fmt = lambda r: f"{'ACEPTADO' if r[0] else 'fallido'} ({r[1]} consultas)"
        p(f"| {v} | {fmt(r1)} | {fmt(r2)} | {fmt(r3)} |")
    p("\nCon bloqueo, a partir de la 5a consulta fallida el id queda bloqueado 5 minutos y "
      "verificar_t devuelve siempre (False, 0.0), asi que el resto de consultas no dan informacion.")
    A["hill_climbing"] = hc

    # 6.4 plantilla "maestra"
    p("\n### 6.4 Template 'maestro' (media de todos los templates de la BD)\n")
    media = T.mean(0)
    sc_m = T @ (media / np.linalg.norm(media))
    A["maestro"] = {"aceptado_por": int((sc_m >= U).sum()), "score_max": float(sc_m.max())}
    p(f"Score maximo contra la BD = {sc_m.max():.3f}; aceptado por {int((sc_m >= U).sum())} identidades. "
      f"(Ademas, sin la clave el atacante no podria calcular esta media en el espacio correcto.)")

    # 6.5 escritura en la BD
    p("\n### 6.5 El atacante ESCRIBE en la base de datos (sin la clave)\n")
    p("Cada fila va cifrada con AES-256-GCM (el id como dato asociado) y la BD lleva un sello HMAC.\n")
    victima = ids[0]
    lineas = bd_orig.read_text(encoding="utf-8").splitlines(keepends=True)
    fila_atacante = lineas[2]  # fila (firmada) de otra persona inscrita
    resto_atacante = fila_atacante.split(",", 1)[1]
    ataques_bd = {
        "anadir fila cifrada con otra clave": "".join(lineas) + f"{victima},{base64.b64encode(os.urandom(12)).decode()},"
                                             f"{base64.b64encode(os.urandom(2064)).decode()}\n",
        "copiar fila firmada de otro con el id de la victima": "".join(lineas) + f"{victima},{resto_atacante}",
        "sustituir la fila de la victima": "".join(l if not l.startswith(victima + ",") else f"{victima},{resto_atacante}" for l in lineas),
        "borrar una fila (DoS)": "".join(lineas[:1] + lineas[2:]),
    }
    A["escritura"] = {}
    for n, contenido in ataques_bd.items():
        bd_tmp = OUT / "bd_ataque.csv"
        copiar_bd(bd_orig, bd_tmp)
        bd_tmp.write_text(contenido, encoding="utf-8", newline="")
        try:
            from sistema_biometrico.base_datos import BaseDatos
            BaseDatos(bd_tmp, s._kh, s._kc)
            r = "CARGADA (fallo)"
        except ErrorIntegridad as e:
            r = f"rechazada: {str(e).split(':', 1)[1].strip()[:90]}"
        A["escritura"][n] = r
        p(f"- {n}: **{r}**")
        borrar_bd(bd_tmp)
    # rollback: el atacante guarda una copia de la BD y la restaura despues de un cambio legitimo
    from sistema_biometrico.base_datos import BaseDatos as _BD
    bd_rb, ver_rb = OUT / "bd_rollback.csv", OUT / "bd_rollback.version"
    copiar_bd(bd_orig, bd_rb)
    ver_rb.unlink(missing_ok=True)
    b = _BD(bd_rb, s._kh, s._kc, ruta_version=ver_rb)
    copia = (bd_rb.read_text(encoding="utf-8"), Path(str(bd_rb) + ".sello").read_text())
    b.eliminar(victima)
    b.guardar()  # cambio legitimo (p.ej. se da de baja a la victima)
    bd_rb.write_text(copia[0], encoding="utf-8", newline="")
    Path(str(bd_rb) + ".sello").write_text(copia[1])
    try:
        _BD(bd_rb, s._kh, s._kc, ruta_version=ver_rb)
        r = "CARGADA (fallo)"
    except ErrorIntegridad as e:
        r = f"rechazada: {str(e).split(':', 1)[1].strip()[:90]}"
    A["escritura"]["restaurar copia antigua completa (rollback)"] = r
    p(f"- restaurar una copia antigua completa, CSV + sello (rollback): **{r}**")
    # borrar la BD entera para empezar vacia y volver a inscribir el id de la victima con otra cara
    borrar_bd(bd_rb)
    try:
        _BD(bd_rb, s._kh, s._kc, ruta_version=ver_rb)
        r = "ARRANCA VACIA (fallo)"
    except ErrorIntegridad as e:
        r = f"rechazada: {str(e).split(':', 1)[1].strip()[:90]}"
    A["escritura"]["borrar la BD entera (CSV + sello)"] = r
    p(f"- borrar la BD entera, CSV + sello, para reinscribir a la victima: **{r}**")
    ver_rb.unlink(missing_ok=True)
    p("\nEl sistema falla cerrado: con la BD manipulada no arranca. Borrar filas sigue siendo una "
      "denegacion de servicio, pero ya no pasa desapercibido.")

    # 6.6 validacion de entradas
    p("\n### 6.6 Entradas malformadas\n")
    for n, t in {"NaN": np.full(512, np.nan), "longitud 128": np.ones(128), "vector nulo": np.zeros(512),
                 "texto": "hola"}.items():
        try:
            s.verificar_t(ids[0], t)
            r = "ACEPTADO (fallo)"
        except Exception as e:
            r = f"rechazado: {type(e).__name__}: {e}"
        p(f"- template {n}: {r}")
    for n, im in {"imagen float en [0,1]": np.asarray(Image.open(fotos[0]).convert("RGB"), dtype=np.float32) / 255,
                  "imagen 10x10": np.zeros((10, 10, 3), np.uint8)}.items():
        try:
            r = s.identificar(im)
        except Exception as e:
            r = f"rechazada: {type(e).__name__}: {e}"
        p(f"- {n}: {r}")

    # 6.7 enumeracion de ids
    sd = sistema_defensas(max_intentos=0)
    r_no = sd.verificar_t("persona_que_no_existe", T[0])
    r_si = sd.verificar_t(ids[1], s._transf.R.T @ T[0])  # impostor real contra un id que existe
    A["enumeracion"] = {"id_inexistente": r_no, "id_existente_impostor": r_si}
    p(f"\n### 6.7 Enumeracion de identidades\n\nverificar_t con un id inexistente devuelve {r_no}; "
      f"con un id inscrito y un impostor devuelve {r_si}. Antes el primero devolvia score=None.")

    # 6.8 bloqueo
    p("\n### 6.8 Bloqueo por intentos fallidos\n")
    sb = sistema_defensas(max_intentos=5)
    genuino = s._transf.R.T @ T[ids.index(victima)]  # template en claro de la victima
    imp_t = s._transf.R.T @ T[1]
    seq = [sb.verificar_t(victima, imp_t)[0] for _ in range(5)] + [sb.verificar_t(victima, genuino)[0]]
    A["bloqueo"] = seq
    p(f"5 intentos con un impostor y despues el template genuino de {victima}: {seq}. Tras el 5o "
      f"fallo, incluso el genuino se rechaza durante 5 minutos.")

    # 6.9 varias caras
    p("\n### 6.9 Fotos con varias caras (las gemelas juntas)\n")
    A["varias_caras"] = {}
    for f in sorted((RAIZ / "data" / "descargas").glob("*")) if (RAIZ / "data" / "descargas").exists() else []:
        if f.suffix.lower() not in EXT:
            continue
        n_caras = len([c for c in s.detector.detectar(s._cargar_imagen(f)) if c["score"] >= 0.8])
        if n_caras < 2:
            continue
        r = s.identificar(f)
        A["varias_caras"][f.name] = r
        p(f"- {f.name} ({n_caras} caras): {'rechazada (ambigua)' if r[1] is None else r}")

    # 6.10 inscripcion
    p("\n### 6.10 Abusar de la inscripcion\n")
    si = sistema_defensas(max_intentos=0)
    os.environ.pop("SB_CLAVE_ADMIN", None)
    victima_i = id_desde_fichero(fotos[0])
    fotos_paula = ([f for f in fotos if "paula" in f.name.lower()]
                   + sorted((RAIZ / "data" / "extra" / (id_p or "-")).glob("*")))
    f_v = next((f for f in fotos if "keanu" in f.name), fotos[0])  # victima con una sola cara en la foto
    im_v = ImageOps.exif_transpose(Image.open(f_v)).convert("RGB")
    variantes_victima = [ImageOps.mirror(im_v), im_v.filter(ImageFilter.GaussianBlur(2)), _jpeg(im_v, 30)]
    for n, fn in {
        "inscribir la cara de una victima con un id nuevo (1 foto)": lambda: si.inscribir("atacante", fotos[0]),
        f"inscribirse con 3 fotos de {id_desde_fichero(f_v)} (espejo, desenfoque, JPEG) con un id nuevo": lambda: si.inscribir("atacante", variantes_victima),
        "inscribir con 3 fotos de personas distintas": lambda: si.inscribir("atacante", fotos[1:4]),
        "reemplazar la identidad de la victima sin clave de admin": lambda: si.inscribir(victima_i, fotos[1], reemplazar=True),
        "anadir una foto propia a la identidad de la victima sin clave de admin": lambda: si.inscribir(victima_i, fotos[1], anadir=True),
    }.items():
        try:
            fn()
            r = "PERMITIDO (fallo)"
        except Exception as e:
            r = f"rechazado: {type(e).__name__}: {e}"
        A.setdefault("inscripcion", {})[n] = r
        p(f"- {n}: {r}")

    # Flujo legitimo de un gemelo: Paula se inscribe en una BD donde ya esta Aitana
    if id_p and len(fotos_paula) >= 3:
        p("\nFlujo legitimo de un gemelo (Paula, con Aitana ya inscrita):\n")
        for n, fs in {"con 1 foto": fotos_paula[:1], f"con sus {len(fotos_paula)} fotos": fotos_paula}.items():
            sg = sistema_defensas(max_intentos=0)
            sg.bd.eliminar(id_p)
            try:
                sg.inscribir(id_p, fs)
                r = "inscrita"
            except Exception as e:
                r = f"rechazada: {e}"
            A.setdefault("inscripcion_gemela", {})[n] = r
            p(f"- {n}: **{r}**")
    for b in (bd_def,):
        borrar_bd(b)
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
    borrar_bd(bd_eval)
    print("\nInforme en", OUT / "informe.md")


if __name__ == "__main__":
    main()
