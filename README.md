# Sistema biométrico facial — CVLFace ViT-KPRPE

Práctica de Sistemas Biométricos. Sistema de reconocimiento facial con **CVLFace ViT-Base KPRPE**
(AdaFace, entrenado en WebFace4M). La persona que elige el grupo es **Paula Etxeberria**, la otra gemela
de *Twin Melody*; su hermana **Aitana** está en el dataset común. El `Dataset.tar.gz` recibido trae
49 fotos, no 50, así que ahora hay **50 personas inscritas**. Para la 51ª basta con añadir la foto que
falta a `data/Dataset/` y volver a ejecutar la inscripción. Las fuentes de las fotos de las gemelas están en [data/FUENTES.md](data/FUENTES.md).

## Pipeline

```
foto ─► [DETECTOR] YuNet (OpenCV)        cara principal → recorte cuadrado (1.6× la caja) a 256×256
     ─► [EXTRACTOR] DFA MobileNet (CVLFace) 5 landmarks → cara alineada 112×112 + keypoints
                    ViT-Base KPRPE          (cara, keypoints) → 512 floats; media con la imagen
                                            en espejo, normalizada L2  = TEMPLATE
     ─► [COMPARADOR] similitud coseno con los templates de la BD (CSV)
```

* **Detector**: YuNet. Si hay varias caras, se queda con la más grande de las que tienen confianza ≥ 0.8.
  El alineador DFA no sirve como detector: con caras pequeñas en la foto falla (Keanu Reeves: confianza 0.06).
* **KPRPE** (*KeyPoint Relative Position Encoding*) usa los 5 landmarks dentro de la atención del ViT,
  por eso el alineador forma parte del extractor.
* **Umbral**: 0.40 de similitud coseno para verificación e identificación. Ver [Evaluación](#evaluación).

## Métodos (`sistema_biometrico.SistemaBiometrico`)

| | con detector | sin detector (`_sd`) | solo comparador (`_t`) |
|---|---|---|---|
| inscribir | `inscribir(id, foto)` | `inscribir_sd(id, foto)` | `inscribir_t(id, template)` |
| verificar | `verificar(id, foto)` → `(bool, score)` | `verificar_sd(id, foto)` | `verificar_t(id, template)` |
| identificar | `identificar(foto)` → `(id \| None, score)` | `identificar_sd(foto)` | `identificar_t(template)` |

* `foto`: ruta, `PIL.Image` o `np.ndarray` RGB `uint8`.
* `template`: cualquier secuencia de 512 números. Se normaliza internamente (L2).
* `_sd` asume que la foto **ya es un recorte de la cara**, centrado: se rellena a cuadrado y se pasa al alineador.
* `inscribir*` lanza `ValueError` si el id ya existe; para sobrescribir, `reemplazar=True`.
  Lanza `CaraNoDetectada` si no encuentra cara.
* `verificar`/`identificar` con detector devuelven `(False, None)` / `(None, None)` si no hay cara.
* `identificar*` devuelve `None` si el mejor score no llega al umbral (persona desconocida).

```python
from sistema_biometrico import SistemaBiometrico
s = SistemaBiometrico("bd/base_datos.csv")
s.verificar("keanu_reeves", "foto.jpg")        # (True, 0.71)
s.identificar("foto.jpg")                      # ('keanu_reeves', 0.71)
t = s.extraer_template("foto.jpg")             # np.float32[512], para los métodos _t
s.identificar_t(t)
```

## Base de datos (`bd/base_datos.csv`)

`id,template,firma`: una fila por inscripción. El template va como 512 floats separados por `;`.
Si la variable de entorno `SB_CLAVE_HMAC` está definida, cada fila se firma con HMAC-SHA256 y al cargar
**se descartan las filas con firma inválida** (avisando con un warning). Sin la variable, no se comprueba la integridad.

## Instalación y uso

### Local (Python 3.12)

```powershell
py -3.12 -m venv .venv
```
```powershell
.venv\Scripts\pip install -r requirements.txt
```
```powershell
.venv\Scripts\python scripts\descargar_modelos.py
```
```powershell
.venv\Scripts\python main.py identificar ruta\a\foto.jpg
```

`descargar_modelos.py` baja unos 470 MB de pesos a `pretrained/` y **parchea** un fichero de CVLFace
(`models/vit_kprpe/RPE/__init__.py`). El original intenta compilar una extensión C++ con
`setup.py install --user` y, cuando falla, deja cambiado el directorio de trabajo, lo que rompe la carga
de pesos. Con el parche se usa la implementación en PyTorch puro, con el mismo resultado en inferencia.

### Docker

```powershell
docker build -t sb-kprpe .
```
```powershell
docker run --rm -v "${PWD}/fotos:/fotos" sb-kprpe verificar keanu_reeves /fotos/cara.jpg
```

CLI: `main.py {inscribir|verificar|identificar|template|listar} ... [--sd] [--bd ruta.csv]`.

### Reconstruir la BD y evaluar

```powershell
.venv\Scripts\python scripts\inscribir_dataset.py
```
```powershell
.venv\Scripts\python scripts\evaluar.py
```

* `inscribir_dataset.py` inscribe `data/Dataset/*` y `data/propia/*`. El id es el nombre del fichero sin el año.
* `evaluar.py` escribe `resultados/informe.md`, `resultados/metricas.json` y `resultados/distribuciones.png`.
  Si existen, usa también `data/test/<id>/*.jpg` (más fotos de personas inscritas, para pares genuinos reales)
  y `data/desconocidos/*.jpg` (personas no inscritas).

## Evaluación

Resultados completos en [resultados/informe.md](resultados/informe.md). Resumen:

* **Impostores** (las 50 personas inscritas comparadas entre sí, 1225 pares): media 0.007, p99 0.178.
  El único par que supera el umbral es el de las gemelas (0.678). Sin ellas, el máximo es 0.343
  (Salma Hayek vs Tini) y FAR = 0 con el umbral de 0.40.
* **Robustez**: cada foto de inscripción, degradada, comparada con su propio template. La estimación es
  optimista porque la foto de origen es la misma. Se identifica bien el 98–100 % con espejo, rotación de
  ±30°, oscuridad, sobreexposición, escala de grises, JPEG q=10, ruido, desenfoque, gafas de sol y
  mascarilla. Falla con la cara rotada 90° y con fotos de 100 px.
* **Gemelas** (`data/test/`, 2–3 fotos de cada una con su nombre):
  * La similitud entre los templates de Paula y de Aitana es **0.678**, más que cualquier par de famosos distintos.
  * Las 5 fotos de prueba se verifican correctamente con su propio id (0.47–0.73)…
  * …pero **las 5 también se aceptan con el id de la hermana** (0.44–0.66). FAR entre gemelas = 100 %.
  * En identificación, 1 de las 5 fotos (una de Aitana) sale como Paula.
  * La diferencia entre "es ella" y "es su hermana" está en torno a ±0.05, por debajo del ruido entre fotos.
    Ningún umbral las separa sin rechazar también a la persona legítima.
* **Pares genuinos reales de otros famosos**: hacen falta más fotos en `data/test/<id>/`. Con solo las gemelas,
  el EER/FAR de la sección 3 del informe mide sobre todo la confusión entre ellas.

## Defectos y ataques conocidos

| # | Defecto / ataque | Resultado | Mitigación / comentario |
|---|---|---|---|
| 1 | **Gemelas** (Paula vs Aitana) | 5/5 fotos aceptadas como la hermana; 1/5 mal identificada | Ningún sistema facial 2D distingue bien a gemelos idénticos. Lo mismo vale, previsiblemente, para Mary-Kate Olsen (Ashley) y James Phelps (Oliver), que también están inscritos. |
| 2 | **Presentación** (foto impresa, pantalla, vídeo, máscara) | Se acepta | No hay detección de vida (PAD). |
| 3 | **Lectura de la BD → replay** con `verificar_t`/`identificar_t` | 49/49 aceptados con score 1.0 | Los templates están en claro y equivalen a credenciales. Haría falta protección de templates (cancelable biometrics, cifrado homomórfico). |
| 4 | **Lectura de la BD → inversión del template** | No probado | En la literatura se reconstruyen caras reconocibles a partir de embeddings. |
| 5 | **Escritura en la BD** (fila con el template del atacante bajo el id de la víctima) | Sin HMAC: aceptado (score 1.0). Con HMAC: fila descartada | HMAC solo sirve si la clave no está al alcance del atacante. Borrar filas sigue siendo una denegación de servicio. |
| 6 | **Hill-climbing con el score** de `verificar_t` como oráculo | Aceptado en ~170–320 consultas | Se devuelve el score. Mitigaciones: devolver solo el booleano, limitar consultas. |
| 7 | **Enumeración de ids** | `verificar*` con un id inexistente devuelve score `None` | Revela quién está inscrito. |
| 8 | **`_sd` con imágenes que no son caras** (ruido, negro, gris, degradado) | Rechazadas (score ≤ 0.12) | La vía `_sd` no tiene control de calidad. Fotos completas por `_sd`: solo 23/49 bien identificadas, porque asume un recorte centrado. |
| 9 | **Varias caras en la foto** | Se usa la más grande con confianza ≥ 0.8 | Un atacante puede poner su cara en primer plano junto a la de otro. |
| 10 | **Caras atípicas** (Adam Pearson, neurofibromatosis) | Con algunas degradaciones YuNet no detecta su cara y elige otra cara de la foto | Sesgo del detector con caras poco representadas en los datos de entrenamiento. |
| 11 | **Rotaciones grandes (90°)** y **resolución muy baja** (≤ 100 px) | Fallan | No se prueban varias orientaciones. |
| 12 | **Templates malformados** (NaN, longitud distinta, nulo, texto) | Rechazados con `ValueError` | Cualquier vector de 512 números finitos es válido tras la normalización. |
| 13 | **Ids** | Solo se aceptan `[\w\-.() ]{1,100}` | Evita inyección de comas o saltos de línea en el CSV. |
