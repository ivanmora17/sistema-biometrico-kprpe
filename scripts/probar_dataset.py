"""Prueba de extremo a extremo con las fotos del dataset: las 49 del profesor + la nuestra (Paula).

Usa la API PUBLICA (con detector, control de calidad, margen de ambiguedad y score cuantizado),
igual que la usara el otro grupo. Solo se desactiva el bloqueo por intentos, porque la prueba de
impostores hace miles de verificaciones fallidas a proposito.

  1. Identificacion 1:N de cada foto.
  2. Verificacion 1:1 de cada foto con su propio id (genuino).
  3. Verificacion 1:1 de cada foto con TODOS los demas ids (impostores: 50 x 49).
  4. Lo mismo por la via sin detector (_sd) y con templates (_t).

Escribe resultados/prueba_dataset.md.  Uso:  python scripts/probar_dataset.py [--bd bd/base_datos.csv]
"""
import argparse
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
warnings.filterwarnings("ignore")
from sistema_biometrico import SistemaBiometrico, CaraNoDetectada  # noqa: E402
from inscribir_dataset import fotos_inscripcion, id_desde_fichero  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", default=str(RAIZ / "bd" / "base_datos.csv"))
    a = ap.parse_args()
    s = SistemaBiometrico(a.bd, max_intentos=0)
    fotos = list(fotos_inscripcion())
    ids = s.bd.identidades()
    L = ["# Prueba con el dataset (49 fotos del profesor + Paula Etxeberria)\n",
         f"Personas inscritas: **{len(ids)}**. API publica con todas las defensas salvo el bloqueo por intentos.\n"]

    filas, temps = [], {}
    for f in fotos:
        id_ = id_desde_fichero(f)
        try:
            temps[id_] = s.extraer_template(f)
        except CaraNoDetectada as e:
            filas.append((id_, f.name, f"sin cara valida: {e}", "-", "-"))
            continue
        ident, sc_i = s.identificar(f)
        ok, sc_v = s.verificar(id_, f)
        filas.append((id_, f.name, ident or "NINGUNO", f"{'SI' if ok else 'NO'} ({sc_v})", sc_i))

    bien_i = sum(1 for r in filas if r[0] == r[2])
    bien_v = sum(1 for r in filas if str(r[3]).startswith("SI"))
    L.append("## 1. Identificacion y verificacion genuina (con detector)\n")
    L.append(f"Identificadas correctamente: **{bien_i}/{len(fotos)}**. Verificadas con su id: **{bien_v}/{len(fotos)}**.\n")
    L.append("| id | foto | identificada como | verificada con su id | score identif. |\n|---|---|---|---|---|")
    for r in filas:
        marca = "" if r[0] == r[2] else " ⚠"
        L.append(f"| {r[0]} | {r[1]} | {r[2]}{marca} | {r[3]} | {r[4]} |")

    # impostores 1:1 por la API publica (_t: mismo comparador que verificar, sin repetir la extraccion)
    acept = []
    n = 0
    for real, t in temps.items():
        for otro in ids:
            if otro == real:
                continue
            n += 1
            if s.verificar_t(otro, t)[0]:
                acept.append((real, otro))
    L.append("\n## 2. Impostores 1:1 (cada foto contra todos los demas ids)\n")
    L.append(f"{n} intentos de impostor, **{len(acept)} aceptados** (FAR = {len(acept) / max(n, 1):.3%}).")
    for r, o in acept:
        L.append(f"- foto de {r} aceptada como {o}")

    # via _t: el template de cada foto identifica a su persona
    ok_t = sum(s.identificar_t(t)[0] == i for i, t in temps.items())
    L.append(f"\n## 3. Via de templates (_t)\n\nidentificar_t con el template de cada foto: **{ok_t}/{len(temps)}** correctas.")

    # via _sd con un recorte de la cara (lo que se espera en _sd)
    ok_sd = 0
    for f in fotos:
        id_ = id_desde_fichero(f)
        rgb = s._cargar_imagen(f)
        recorte = s._recortar_con_detector(rgb)  # simula que el usuario pasa ya la cara recortada
        ok_sd += s.identificar_sd(recorte)[0] == id_
    L.append(f"\n## 4. Via sin detector (_sd) con la cara ya recortada\n\nidentificar_sd: **{ok_sd}/{len(fotos)}** correctas.")

    out = RAIZ / "resultados" / "prueba_dataset.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
