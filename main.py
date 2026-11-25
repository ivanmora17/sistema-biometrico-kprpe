"""CLI del sistema.

  python main.py inscribir   <id> <foto>   [--sd]
  python main.py verificar   <id> <foto>   [--sd]
  python main.py identificar <foto>        [--sd]
  python main.py template    <foto>        [--sd]   -> imprime el template (para *_t)
  python main.py listar
"""
import argparse
import warnings

warnings.filterwarnings("ignore")
from sistema_biometrico import SistemaBiometrico  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["inscribir", "verificar", "identificar", "template", "listar"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--sd", action="store_true", help="omitir el detector de caras")
    ap.add_argument("--bd", default="bd/base_datos.csv")
    a = ap.parse_args()
    s = SistemaBiometrico(a.bd)
    suf = "_sd" if a.sd else ""
    if a.accion == "inscribir":
        getattr(s, "inscribir" + suf)(a.args[0], a.args[1])
        print("inscrito", a.args[0])
    elif a.accion == "verificar":
        ok, score = getattr(s, "verificar" + suf)(a.args[0], a.args[1])
        print(f"{'ACEPTADO' if ok else 'RECHAZADO'}  score={score}  umbral={s.umbral_verificacion}")
    elif a.accion == "identificar":
        id_, score = getattr(s, "identificar" + suf)(a.args[0])
        print(f"{id_ or 'DESCONOCIDO'}  score={score}  umbral={s.umbral_identificacion}")
    elif a.accion == "template":
        print(";".join(f"{v:.6f}" for v in s.extraer_template(a.args[0], con_detector=not a.sd)))
    else:
        for i in s.bd.identidades():
            print(i)
