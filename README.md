# Sistema biométrico facial — CVLFace ViT-KPRPE

Práctica de Sistemas Biométricos. Sistema de reconocimiento facial con **CVLFace ViT-Base KPRPE**
(AdaFace, entrenado en **WebFace12M**; la versión WebFace4M sigue disponible con `SB_KPRPE=webface4m`). La persona que elige el grupo es **Paula Etxeberria**, la otra gemela
de *Twin Melody*; su hermana **Aitana** está en el dataset común. El `Dataset.tar.gz` recibido trae
49 fotos, no 50 (comprobado también en el `Dataset.tar.gz` del campus), así que con Paula hay **50 personas inscritas**. Para la 51ª basta con añadir la foto que
falta a `data/Dataset/` y volver a ejecutar la inscripción. Las fuentes de las fotos de las gemelas están en [data/FUENTES.md](data/FUENTES.md).

## Uso rápido (Docker)

```bash
docker build -t sb-kprpe .
```
```bash
docker run --rm -v "$PWD/fotos:/fotos:ro" sb-kprpe identificar /fotos/cara.jpg
```
```bash
docker run --rm -v "$PWD/fotos:/fotos:ro" sb-kprpe verificar keanu_reeves /fotos/cara.jpg
```

Al construir la imagen se descargan los pesos (versiones fijadas y comprobadas por SHA-256), se
genera una **clave aleatoria propia de esa imagen** y se inscriben las 50 personas. La base de datos
sale cifrada, firmada y con los templates protegidos. El contenedor se ejecuta con un usuario sin privilegios.

Cada `docker run --rm` arranca con la BD de la imagen. Para que las inscripciones nuevas y el
bloqueo por intentos se conserven entre ejecuciones, usa un volumen con nombre, que la primera vez
se rellena con la BD de la imagen:

```bash
docker run --rm -v sb-bd:/app/bd -v "$PWD/fotos:/fotos:ro" sb-kprpe inscribir yo /fotos/yo.jpg
```

### CLI

```
main.py [--bd ruta.csv] inscribir     <id> <foto> [<foto> ...] [--sd] [--reemplazar | --anadir] [--admin CLAVE]
main.py [--bd ruta.csv] verificar     <id> <foto> [--sd]
main.py [--bd ruta.csv] identificar   <foto>      [--sd]
main.py [--bd ruta.csv] template      <foto>      [--sd]     # 512 valores separados por ';'
main.py [--bd ruta.csv] verificar_t   <id> <fichero|->       # template de un fichero o de stdin
main.py [--bd ruta.csv] identificar_t <fichero|->
main.py [--bd ruta.csv] listar
```

El resultado sale por stdout y los avisos y errores por stderr, así que
`... template foto.jpg > t.txt` deja en `t.txt` solo el template. Para leer de stdin con Docker hace
falta `-i`: `docker run --rm -i sb-kprpe verificar_t adele - < t.txt`.
Código de salida: 0 = aceptado/identificado, 1 = rechazado/desconocido/sin cara válida, 2 = error.

## Pipeline

```
foto ─► [CALIDAD]   tamaño ≤ 40 Mpx, se reduce a 2048 px; si es pequeña se amplía a 640 px para detectar
     ─► [DETECTOR]  YuNet (OpenCV)           cara principal → recorte cuadrado (1.6× la caja) a 256×256
                                             si la foto viene girada, prueba 90/180/270° y se queda con la mejor
                                             se rechaza si hay 2 caras claras de tamaño parecido o la cara mide < 16 px
     ─► [EXTRACTOR] DFA MobileNet (CVLFace)  5 landmarks → cara alineada 112×112 + keypoints
                                             se rechaza si la confianza del alineador es < 0.5
                    ViT-Base KPRPE           (cara, keypoints) → 512 floats; media con la imagen
                                             en espejo, normalizada L2  = TEMPLATE
     ─► [COMPARADOR] similitud coseno con los templates (protegidos) de la BD
                     acepta si score ≥ 0.40 y ninguna otra identidad inscrita queda a menos de 0.03
```

* **Detector**: YuNet. Si hay varias caras, se queda con la más grande de las que tienen confianza ≥ 0.8,
  salvo que haya otra cara clara de al menos la mitad de área: entonces la foto es ambigua y se rechaza.
* **KPRPE** (*KeyPoint Relative Position Encoding*) usa los 5 landmarks dentro de la atención del ViT,
  por eso el alineador forma parte del extractor.
* **Umbral**: 0.40 de similitud coseno para verificación e identificación, con un **margen de
  ambigüedad** de 0.03. Ver [Evaluación](#evaluación).

## Métodos (`sistema_biometrico.SistemaBiometrico`)

| | con detector | sin detector (`_sd`) | solo comparador (`_t`) |
|---|---|---|---|
| inscribir | `inscribir(id, foto \| [fotos])` | `inscribir_sd(id, foto \| [fotos])` | `inscribir_t(id, template \| [templates])` |
| verificar | `verificar(id, foto)` → `(bool, score)` | `verificar_sd(id, foto)` | `verificar_t(id, template)` |
| identificar | `identificar(foto)` → `(id \| None, score)` | `identificar_sd(foto)` | `identificar_t(template)` |

* `foto`: ruta, `PIL.Image` o `np.ndarray` RGB (`uint8`, o `float` en [0, 1]).
* `template`: cualquier secuencia de 512 números (la que da `extraer_template` o `main.py template`). Se normaliza internamente (L2).
* `score`: similitud coseno **redondeada hacia abajo a 0.1** y nunca negativa. `None` = la foto no tiene
  una cara utilizable (ninguna, varias o de mala calidad).
* `identificar*` devuelve `None` si el mejor score no llega al umbral (desconocido) o si la segunda
  identidad también lo supera y está a menos de 0.03 (ambigua; la CLI lo indica).
* `_sd` asume que la foto **ya es un recorte de la cara**, centrado: se rellena a cuadrado y se pasa al alineador.
* `inscribir*` acepta **una foto o varias** de la misma persona. Una basta en el caso normal. Si la cara
  se parece a alguien ya inscrito (gemelo o doble: score ≥ umbral), hacen falta **al menos 3 fotos**, y en
  2/3 de ellas la cara tiene que parecerse más a sus otras fotos que a esa persona (por el margen); si no,
  el sistema no podría distinguirlas y se rechaza (`ValueError`). Las fotos de una inscripción deben ser de
  la misma persona. Lanza `ValueError` si el id ya existe, `PermissionError` si se intenta reemplazar o
  añadir fotos (`anadir=True`) sin la clave de administrador, y `CaraNoDetectada` / `CaraNoValida` si una
  foto no sirve.

```python
from sistema_biometrico import SistemaBiometrico
s = SistemaBiometrico("bd/base_datos.csv")
s.verificar("keanu_reeves", "foto.jpg")        # (True, 0.7)
s.identificar("foto.jpg")                      # ('keanu_reeves', 0.7)
t = s.extraer_template("foto.jpg")             # np.float32[512], para los métodos _t
s.identificar_t(t)
```

## Seguridad

| # | Medida | Contra qué | Detalle |
|---|---|---|---|
| 1 | **BD cifrada y autenticada** | Leer o escribir la BD | Cada template va cifrado con **AES-256-GCM** (nonce aleatorio, el id como dato asociado): el CSV no contiene ningún vector legible, y el tag detecta filas cambiadas, añadidas sin la clave o movidas a otro id. La BD entera lleva además un sello HMAC (`base_datos.csv.sello`) que detecta filas borradas o duplicadas. El sello lleva además un número de versión, y la última versión vista se guarda firmada fuera de `bd/` (junto a la clave): restaurar una copia antigua completa (CSV + sello) o borrar la BD entera también se detecta. Si algo no cuadra, el sistema **no arranca** (`ErrorIntegridad`): falla cerrado. |
| 2 | **Templates cancelables** | Reinyectar templates robados (*replay*) | Lo que se cifra es R·t, con R una rotación ortogonal secreta de 512×512 derivada de la clave. El coseno se conserva (misma precisión), pero una fila robada no sirve en `verificar_t`, que vuelve a aplicar R. Cambiar la clave revoca todos los templates. |
| 3 | **Score cuantizado** (0.1, ≥ 0) | Hill-climbing con el score como oráculo | Sin resolución fina no hay gradiente que seguir. |
| 4 | **Bloqueo por intentos** | Hill-climbing, fuerza bruta | 5 verificaciones fallidas seguidas contra un id → bloqueado 5 min (`SB_MAX_INTENTOS`, `SB_BLOQUEO_SEG`; 0 lo desactiva). Cuenta también ids inexistentes. Se guarda en `bd/base_datos.intentos.json`, con los ids como HMAC. |
| 5 | **Sin enumeración de ids** | Averiguar quién está inscrito | Un id inexistente devuelve `(False, 0.0)`, igual que un impostor. |
| 6 | **Margen de ambigüedad** | Gemelos, parecidos | Las gemelas están inscritas con **varias fotos cada una** (Paula 3, Aitana 4) y se compara con la más parecida. Además se rechaza si otra identidad inscrita supera el umbral y queda a menos de 0.03 de la reclamada (1:1) o de la primera (1:N). |
| 7 | **Control de calidad** | Fotos con varias caras, no-caras por `_sd`, caras diminutas | Una sola cara clara, ≥ 16 px, alineación con confianza ≥ 0.5. Imágenes > 40 Mpx rechazadas (bombas de descompresión). |
| 8 | **Inscripción protegida** | Suplantar o duplicar identidades; colar a un gemelo | Una cara que se parece a alguien inscrito solo entra con ≥ 3 fotos que el sistema pueda distinguir de esa persona (política general, no un caso especial para las gemelas). Quien intenta inscribirse con fotos de una persona inscrita no pasa esa comprobación. Reemplazar o añadir fotos a una identidad exige `SB_CLAVE_ADMIN` (sin ella está deshabilitado). Si se define, también hace falta para inscribir. |
| 9 | **Cadena de suministro** | Pesos cambiados o manipulados | Repos de HuggingFace y YuNet fijados a un commit, con los pesos comprobados por SHA-256 al descargar. |

**Clave.** La clave maestra (32 bytes) se lee de `SB_CLAVE` (hex) o del fichero `SB_CLAVE_FICHERO`
(por defecto `secreto/clave`). Si no existe, se genera. Está fuera de `bd/`, fuera de git y fuera
del contexto de Docker. De ella se derivan tres subclaves independientes: cifrado, HMAC y rotación.

**Límite (importante).** Todo se ejecuta en la máquina de quien usa el sistema. Quien controla el
proceso (tiene el código, la imagen y por tanto la clave) puede leer `secreto/clave`, descifrar la BD,
deshacer R, re-firmarla o simplemente quitar cualquier comprobación del código. Estas medidas protegen frente
a quien **usa la API/CLI** o **solo tiene acceso a la BD**, no frente a quien controla el ordenador.
Para eso haría falta separar el comparador en un servidor (o en hardware seguro) al que el usuario
solo llega por la API. Además, las fotos de inscripción son públicas (están en el repo), así que
cualquiera puede sacar el template de una persona inscrita a partir de su foto. El commit inicial
del repo incluía la BD con los templates en claro, y siguen siendo válidos por la misma razón.

## Instalación local (Python 3.12)

Linux / macOS:

```bash
python3.12 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
```bash
.venv/bin/python scripts/descargar_modelos.py
```
```bash
.venv/bin/python scripts/inscribir_dataset.py
```
```bash
.venv/bin/python main.py identificar ruta/a/foto.jpg
```

Windows (PowerShell): igual, pero con `py -3.12 -m venv .venv` y `.venv\Scripts\python ...`.

`inscribir_dataset.py` crea la clave (si no existe) y la BD firmada. La BD no está en git porque
depende de la clave. `descargar_modelos.py` baja unos 470 MB de pesos a `pretrained/` y **parchea** un
fichero de CVLFace (`models/vit_kprpe/RPE/__init__.py`). El original intenta compilar una extensión C++
con `setup.py install --user` y, cuando falla, deja cambiado el directorio de trabajo, lo que rompe la
carga de pesos. Con el parche se usa la implementación en PyTorch puro, con el mismo resultado en
inferencia. Por eso al cargar aparece `Failed to import cuda/cpp RPEIndexFunction` en stderr: es normal.

### Reconstruir la BD y evaluar

```bash
python scripts/inscribir_dataset.py
```
```bash
python scripts/evaluar.py
```

Con Docker, montando los datos y una carpeta de resultados escribible:

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/resultados:/app/resultados" --entrypoint python sb-kprpe scripts/evaluar.py
```

* `inscribir_dataset.py` inscribe `data/Dataset/*` y `data/propia/*` (el id es el nombre del fichero sin el año) y añade
  como templates extra las fotos de `data/extra/<id>/` (3 más de Aitana y 2 más de Paula).
  Usa la API pública con la misma política que cualquier usuario: Paula se inscribe con sus 3 fotos porque
  se parece a Aitana, y pasa la comprobación de que el sistema las distingue.
* `evaluar.py` escribe `resultados/informe.md`, `resultados/metricas.json` y `resultados/distribuciones.png`.
  Las métricas usan el score sin cuantizar y sin bloqueo; los ataques usan la API pública con las defensas.
  Si existen, usa también `data/test/<id>/*.jpg` (más fotos de personas inscritas, para pares genuinos reales)
  y `data/desconocidos/*.jpg` (personas no inscritas).

## Evaluación

### Prueba con el dataset (las 49 del profesor + Paula)

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/resultados:/app/resultados" --entrypoint python sb-kprpe scripts/probar_dataset.py
```

Con la API pública y todas las defensas (salvo el bloqueo, porque se hacen miles de intentos fallidos a propósito).
Resultado en [resultados/prueba_dataset.md](resultados/prueba_dataset.md):

* Identificación: **50/50** correctas. Verificación con su propio id: **50/50**.
* Impostores 1:1 (cada foto contra los otros 49 ids, 2450 intentos): **0 aceptados**, gemelas incluidas.
* Vía `_t` (templates): 50/50. Vía `_sd` con la cara ya recortada: 50/50.

Las fotos de prueba son las mismas que se inscriben, así que esto comprueba que el sistema funciona de
extremo a extremo, no cuánto acierta con fotos nuevas.

### Evaluación completa

Resultados en [resultados/informe.md](resultados/informe.md). Resumen:

* **Impostores** (las 50 personas inscritas comparadas entre sí, 1225 pares): media 0.004, p99 0.137.
  El único par que supera el umbral es el de las gemelas (0.642). Sin ellas, el máximo es 0.266
  (Salma Hayek vs Tini). Con el extractor WebFace4M eran p99 0.178 y máximo 0.343: el de 12M separa mejor.
* **Robustez**: cada foto de inscripción, degradada, comparada con su propio id. La estimación es
  optimista porque la foto de origen es la misma. Se acepta el **98–100 %** con espejo, rotación de
  15°, 30° y **90°** (foto girada), oscuridad, sobreexposición, escala de grises, JPEG q=10, ruido,
  desenfoque, gafas de sol, mascarilla y fotos de 200 px, y el 90 % con fotos de 100 px. El fallo
  que se repite es Adam Pearson (neurofibromatosis): YuNet le da poca confianza a su cara.
* **Gemelas** — el ataque obvio es usar una foto de Aitana para pasar por Paula, o al revés:
  * Similitud entre las fotos de inscripción de Paula y Aitana: **0.642**, más que cualquier par de famosos distintos.
  * Con una foto por persona y solo el umbral (sistema original), **todas** las fotos de una gemela se
    aceptaban como la hermana (FAR 100 %).
  * Ahora cada gemela está inscrita con varias fotos etiquetadas (`data/extra/`) y hay margen de ambigüedad.
    Probado con **leave-one-out** (cada foto contra la BD sin su propio template, como si fuera nueva) sobre
    las 7 fotos etiquetadas: **ninguna se acepta como la hermana (FAR 0 %)** y **6 de 7 se aceptan como
    ellas mismas** e identifican bien. La que falla (la foto de inscripción de Paula) sale como *ambigua*:
    se rechaza, no se confunde.
  * Comparado: con una sola foto por gemela y margen 0.05 se aceptaban 3/5; un centroide o una dirección
    discriminante entre las dos daban peores resultados (1–3 de 7 aceptadas como la hermana).
  * Son solo 7 fotos; con más fotos etiquetadas de cada gemela el sistema las separaría todavía mejor.
* **Pares genuinos reales de otros famosos**: no hay fotos de prueba distintas de las de inscripción
  (`data/test/<id>/`) ni de personas no inscritas (`data/desconocidos/`). El FRR con fotos nuevas
  (otra edad, pose o iluminación) no está medido.

## Ataques probados

| # | Ataque | Antes | Ahora |
|---|---|---|---|
| 1 | **Gemelas** (Paula vs Aitana) | Todas aceptadas como la hermana | 0/7 como la hermana; 6/7 aceptadas como ellas mismas (leave-one-out) |
| 2 | **Presentación** (foto impresa, pantalla, vídeo, máscara) | Se acepta | Se acepta: no hay detección de vida (PAD). Se probó Silent-Face-Anti-Spoofing (MiniFASNet) y se descartó, ver [Detección de vida](#detección-de-vida-descartada). |
| 3 | **Leer la BD → replay** con `verificar_t`/`identificar_t` | 50/50 aceptados | El CSV está cifrado; incluso con los templates descifrados, 0/50 (templates cancelables) |
| 4 | **Leer la BD → inversión del template** | Posible en teoría | Hace falta la clave para descifrar y para deshacer R |
| 5 | **Escribir en la BD** (añadir, sustituir, copiar o borrar filas) | Aceptado sin HMAC (score 1.0) | El sistema no arranca (`ErrorIntegridad`), también si se restaura una copia antigua completa (rollback) o se borra la BD entera para reinscribir a alguien con otra cara. |
| 6 | **Hill-climbing** con el score de `verificar_t` | Aceptado en ~200–340 consultas | 0/5 en 5000 consultas con score cuantizado; bloqueado a la 5ª con bloqueo |
| 7 | **Enumeración de ids** | id inexistente → score `None` | `(False, 0.0)`, igual que un impostor |
| 8 | **`_sd` con imágenes que no son caras** | Rechazadas por score (≤ 0.12) | Rechazadas por el control de alineación |
| 9 | **Varias caras en la foto** | Se usaba la más grande | Rechazada si hay dos caras claras parecidas (4/4 fotos de las gemelas juntas) |
| 10 | **Inscribir la cara de otro** con un id nuevo (1 foto o 3 variantes), con fotos de personas distintas, o **reemplazar / añadir fotos** a su identidad | Permitido | Rechazado (parecido sin fotos suficientes, no distinguible, fotos de distintas personas, falta clave de admin). Un gemelo legítimo con 3 fotos distinguibles sí se inscribe (Paula). |
| 11 | **Caras atípicas** (Adam Pearson, neurofibromatosis) | Con algunas degradaciones YuNet elegía otra cara | Si la otra cara es de tamaño parecido, la foto se rechaza como ambigua; si es más grande y clara, se sigue comparando la cara equivocada. Sesgo del detector |
| 12 | **Templates o imágenes malformados** (NaN, longitud distinta, nulo, texto, float, 10×10, > 40 Mpx) | Rechazados | Rechazados; los float en [0, 1] ahora se escalan bien |
| 13 | **Ids** | Solo `[\w\-.() ]{1,100}` | Igual (evita inyección en el CSV) |
| 14 | **Morphing / ejemplos adversarios** | No probado | No probado. Sin defensa específica; el margen de ambigüedad ayuda algo contra un morph de dos personas inscritas. |
| 15 | **Quien controla la máquina** | — | Puede saltarse todo (ver Límite) |

## Detección de vida (descartada)

Se midió el detector de ataques de presentación **Silent-Face-Anti-Spoofing** (MiniFASNetV2 + MiniFASNetV1SE,
Apache-2.0) sobre las 55 fotos legítimas que tenemos (las 50 inscritas y las 5 de prueba de las gemelas):

* 53/55 clasificadas como reales (mediana 0.989).
* **Michael Jackson 1982** sale como ataque (0.09; probablemente es el escaneo de una foto impresa) y
  **Cynthia Erivo** queda en la frontera (0.41).

Requisito del grupo: el sistema **no puede fallar con las fotos del profesor**. Con el detector activado
fallaría al menos con Michael Jackson, así que no se integra. Además, el sistema recibe fotos y no una
cámara, y las fotos de la práctica son fotos de internet (muchas, escaneos o fotos de fotos), así que la
frontera entre "foto legítima" y "foto de una foto" no está clara. No se ha podido medir con ataques
reales (fotos de fotos impresas o de pantallas).
