"""CLI del sistema.

  python main.py inscribir   <id> <foto> [<foto> ...] [--sd] [--reemplazar | --anadir] [--admin CLAVE]
  python main.py verificar   <id> <foto>   [--sd]
  python main.py identificar <foto>        [--sd]
  python main.py template    <foto>        [--sd]   -> imprime el template (entrada de *_t)
  python main.py verificar_t <id> <fichero_template>   ('-' = leer de stdin)
  python main.py identificar_t <fichero_template>
  python main.py listar

El resultado sale por stdout; avisos y errores por stderr. Codigo de salida: 0 = aceptado /
identificado / hecho, 1 = rechazado / desconocido / sin cara valida, 2 = error.
"""
import argparse
import sys
import warnings

# Solo se silencian los avisos de librerias de terceros; los del sistema (bloqueos, clave nueva...)
# se muestran por stderr.
warnings.filterwarnings("ignore")
warnings.filterwarnings("default", module=r"sistema_biometrico(\..*)?")
warnings.showwarning = lambda m, c, f, l, file=None, line=None: print(f"aviso: {m}", file=sys.stderr)


def leer_template(ruta):
    with (sys.stdin if ruta == "-" else open(ruta, encoding="utf-8")) as f:
        txt = f.read().strip()
    return [float(v) for v in txt.replace("\n", ";").replace(",", ";").split(";") if v.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sistema biometrico facial (CVLFace ViT-KPRPE)")
    ap.add_argument("--bd", default="bd/base_datos.csv", help="ruta de la base de datos CSV")
    sub = ap.add_subparsers(dest="accion", required=True)

    def p(nombre, *args, sd=True):
        sp = sub.add_parser(nombre)
        for a in args:
            sp.add_argument(a)
        if sd:
            sp.add_argument("--sd", action="store_true", help="omitir el detector (la foto ya es un recorte)")
        return sp

    pi = sub.add_parser("inscribir")
    pi.add_argument("id")
    pi.add_argument("fotos", nargs="+", help="una o varias fotos de la persona (varias si es gemelo/doble de alguien)")
    pi.add_argument("--sd", action="store_true", help="omitir el detector (las fotos ya son recortes)")
    pi.add_argument("--reemplazar", action="store_true")
    pi.add_argument("--anadir", action="store_true", help="anadir fotos a una identidad ya inscrita")
    pi.add_argument("--admin", help="clave de administrador (SB_CLAVE_ADMIN)")
    p("verificar", "id", "foto")
    p("identificar", "foto")
    p("template", "foto")
    p("verificar_t", "id", "template", sd=False)
    p("identificar_t", "template", sd=False)
    p("listar", sd=False)
    a = ap.parse_args(argv)

    from sistema_biometrico import SistemaBiometrico, CaraNoDetectada, ErrorIntegridad
    try:
        s = SistemaBiometrico(a.bd)
        suf = "_sd" if getattr(a, "sd", False) else ""
        if a.accion == "inscribir":
            getattr(s, "inscribir" + suf)(a.id, a.fotos, a.reemplazar, clave_admin=a.admin, anadir=a.anadir)
            print("inscrito", a.id)
            return 0
        if a.accion in ("verificar", "verificar_t"):
            entrada = leer_template(a.template) if a.accion == "verificar_t" else a.foto
            fn = s.verificar_t if a.accion == "verificar_t" else getattr(s, "verificar" + suf)
            ok, score = fn(a.id, entrada)
            estado = "SIN CARA VALIDA" if score is None else ("ACEPTADO" if ok else "RECHAZADO")
            print(f"{estado}  score={score}  umbral={s.umbral_verificacion}")
            return 0 if ok else 1
        if a.accion in ("identificar", "identificar_t"):
            if a.accion == "identificar_t":
                id_, score = s.identificar_t(leer_template(a.template))
            else:
                id_, score = getattr(s, "identificar" + suf)(a.foto)
            if id_:
                res = id_
            elif score is None:
                res = "SIN CARA VALIDA"
            elif score >= s.umbral_identificacion:
                res = "AMBIGUA (se parece casi igual a dos identidades inscritas)"
            else:
                res = "DESCONOCIDO"
            print(f"{res}  "
                  f"score={score}  umbral={s.umbral_identificacion}")
            return 0 if id_ else 1
        if a.accion == "template":
            print(";".join(f"{v:.6f}" for v in s.extraer_template(a.foto, con_detector=not a.sd)))
            return 0
        for i in s.bd.identidades():
            print(i)
        return 0
    except CaraNoDetectada as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ErrorIntegridad as e:
        print(f"ERROR DE INTEGRIDAD: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError, PermissionError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
