# Informe de evaluacion - CVLFace ViT-KPRPE

Personas inscritas: **50** | umbral verificacion = **0.4** | umbral identificacion = **0.4** (similitud coseno) | margen de ambiguedad = **0.03**

Las metricas usan el score real (sin cuantizar) y la regla de decision completa: umbral + rechazo si otra identidad inscrita queda a menos del margen. Ver seccion 6 para las defensas.

## 1. Distribucion de impostores (inscritos entre si)

1225 pares. media=0.004, p99=0.137, max=0.642, pares por encima del umbral = 0.0008

Pares de impostores mas parecidos:

| persona A | persona B | score |
|---|---|---|
| aitana_etxebarria_(twin_melody) | paula_etxeberria_(twin_melody) | 0.642 |
| salma_hayek | tini | 0.266 |
| arianna_grande | taylor_swift | 0.223 |
| cynthia_erivo | josie_totah | 0.161 |
| arianna_grande | samuel_l_jackson | 0.155 |
| brays_efe | lady_gaga | 0.153 |
| idris_elba | salma_hayek | 0.151 |
| forest_whitaker | mindy_kaling | 0.143 |

## 2. Pares genuinos sinteticos (foto de inscripcion transformada)

Cada foto de inscripcion se transforma y se verifica contra su propio id. Es una estimacion OPTIMISTA (misma foto de origen), pero mide la robustez a cada degradacion. Las fotos que el control de calidad rechaza (sin cara, varias caras, cara < 16 px, mala alineacion) cuentan como rechazo en TAR, pero no entran en las columnas de score.

| transformacion | score medio | score min | TAR (verif.) | identif. correcta | sin cara valida |
|---|---|---|---|---|---|
| original | 1.000 | 1.000 | 100% | 100% | 0 |
| espejo | 0.990 | 0.969 | 98% | 98% | 1 |
| rotacion 15 | 0.971 | -0.003 | 98% | 98% | 0 |
| rotacion 30 | 0.986 | 0.938 | 98% | 98% | 1 |
| rotacion 90 | 0.980 | -0.000 | 98% | 98% | 0 |
| baja res 200px | 0.976 | 0.865 | 98% | 98% | 1 |
| baja res 100px | 0.887 | 0.531 | 90% | 90% | 5 |
| desenfoque r=3 | 0.974 | 0.812 | 100% | 100% | 0 |
| oscura x0.4 | 0.992 | 0.977 | 100% | 100% | 0 |
| sobreexpuesta x1.8 | 0.916 | 0.600 | 98% | 98% | 1 |
| escala de grises | 0.952 | 0.871 | 100% | 100% | 0 |
| JPEG q=10 | 0.924 | 0.816 | 98% | 98% | 1 |
| ruido sigma=25 | 0.954 | 0.830 | 100% | 100% | 0 |
| gafas de sol | 0.772 | 0.590 | 98% | 98% | 1 |
| mascarilla | 0.773 | 0.613 | 98% | 98% | 1 |

> No hay data/test/: no se calculan pares genuinos reales (ver README).

## 4. Personas NO inscritas (identificacion en conjunto abierto)

> No hay data/desconocidos/: no se ha probado con personas no inscritas.

## 5. Gemelas Twin Melody (Paula vs Aitana)

Similitud entre las fotos de inscripcion de **paula_etxeberria_(twin_melody)** y **aitana_etxebarria_(twin_melody)**: **0.642** (SUPERA el umbral 0.4). Hay 0 pares de impostores con score mayor.

Cada gemela esta inscrita con varias fotos (Paula: 3, Aitana: 4; ver data/extra). Prueba **leave-one-out**: cada foto etiquetada de una gemela se compara con la BD SIN su propio template, como si fuera una foto nueva. 'Antes' = sistema original (una foto por persona y solo umbral); 'ahora' = varias fotos + margen de ambiguedad 0.03.

| persona real | foto | score su id | score hermana | aceptada como ella | aceptada como hermana (antes) | aceptada como hermana (ahora) | identificada como |
|---|---|---|---|---|---|---|---|
| paula_etxeberria_(twin_melody) | paula_etxeberria_(twin_melody)_2023.jpg | 0.654 | 0.642 | no | SI | no | AMBIGUA |
| paula_etxeberria_(twin_melody) | neo2_1_izda.jpg | 0.599 | 0.555 | SI | SI | no | paula_etxeberria_(twin_melody) |
| paula_etxeberria_(twin_melody) | neo2_2_izda.jpg | 0.654 | 0.566 | SI | SI | no | paula_etxeberria_(twin_melody) |
| aitana_etxebarria_(twin_melody) | aitana_etxebarria_(twin_melody)_2024.jpg | 0.675 | 0.642 | SI | SI | no | aitana_etxebarria_(twin_melody) |
| aitana_etxebarria_(twin_melody) | formulatv.jpg | 0.699 | 0.593 | SI | SI | no | aitana_etxebarria_(twin_melody) |
| aitana_etxebarria_(twin_melody) | neo2_1_dcha.jpg | 0.645 | 0.555 | SI | SI | no | aitana_etxebarria_(twin_melody) |
| aitana_etxebarria_(twin_melody) | neo2_2_dcha.jpg | 0.699 | 0.562 | SI | SI | no | aitana_etxebarria_(twin_melody) |

FAR entre gemelas: **100% antes -> 0% ahora**. Aceptada como ella misma: **86%**; identificacion correcta: 86% (si la foto es ambigua se rechaza a las dos). Solo hay 7 fotos etiquetadas: con mas fotos de cada gemela el sistema mejoraria.

## 6. Ataques y defensas


### 6.1 Saltarse el detector (*_sd) con imagenes que no son caras

| entrada | con detector | sin detector (_sd) |
|---|---|---|
| ruido uniforme | rechazada (sin cara valida) | rechazada (sin cara valida) |
| gris liso | rechazada (sin cara valida) | rechazada (sin cara valida) |
| negro | rechazada (sin cara valida) | rechazada (sin cara valida) |
| degradado | rechazada (sin cara valida) | rechazada (sin cara valida) |

Fotos de inscripcion completas (sin recortar) por identificar_sd: 18/50 bien identificadas, 27 rechazadas por el control de alineacion, 5 identificadas mal o como desconocidas. Antes se aceptaban todas y salian 24/50 bien.

### 6.2 El atacante LEE la base de datos (replay de templates)

El CSV va cifrado (AES-256-GCM): sin la clave no contiene ningun vector. Aqui se supone el peor caso: el atacante consigue los templates guardados ya descifrados (p.ej. leyendo la memoria del proceso) y los reinyecta.

Reinyectando los templates guardados en verificar_t/identificar_t: 0/55 verificaciones aceptadas, 0/55 identificaciones (antes: 50/50). La BD guarda R·t con R secreta; verificar_t aplica R otra vez y R·R·t no se parece a R·t. Quien tenga ademas la clave puede deshacer R (ver README).

### 6.3 Hill-climbing usando verificar_t como oraculo

El atacante NO lee la BD; prueba templates aleatorios y se queda con el que mejora el score (o lo iguala, para poder avanzar sobre las mesetas del score cuantizado).

| victima | score exacto (antes) | score cuantizado | cuantizado + bloqueo |
|---|---|---|---|
| adam_pearson | ACEPTADO (254 consultas) | fallido (5000 consultas) | fallido (200 consultas) |
| adele | ACEPTADO (272 consultas) | fallido (5000 consultas) | fallido (200 consultas) |
| aitana_etxebarria_(twin_melody) | ACEPTADO (276 consultas) | fallido (5000 consultas) | fallido (200 consultas) |
| anya_taylor_joy | ACEPTADO (264 consultas) | fallido (5000 consultas) | fallido (200 consultas) |
| arianna_grande | ACEPTADO (233 consultas) | fallido (5000 consultas) | fallido (200 consultas) |

Con bloqueo, a partir de la 5a consulta fallida el id queda bloqueado 5 minutos y verificar_t devuelve siempre (False, 0.0), asi que el resto de consultas no dan informacion.

### 6.4 Template 'maestro' (media de todos los templates de la BD)

Score maximo contra la BD = 0.522; aceptado por 7 identidades. (Ademas, sin la clave el atacante no podria calcular esta media en el espacio correcto.)

### 6.5 El atacante ESCRIBE en la base de datos (sin la clave)

Cada fila va cifrada con AES-256-GCM (el id como dato asociado) y la BD lleva un sello HMAC.

- anadir fila cifrada con otra clave: **rechazada: linea 57 (id='adam_pearson'): no descifra (cambiada, de otro id o sin la clave)**
- copiar fila firmada de otro con el id de la victima: **rechazada: linea 57 (id='adam_pearson'): no descifra (cambiada, de otro id o sin la clave)**
- sustituir la fila de la victima: **rechazada: linea 2 (id='adam_pearson'): no descifra (cambiada, de otro id o sin la clave)**
- borrar una fila (DoS): **rechazada: se han borrado, duplicado o reordenado filas, o falta bd_ataque.csv.sello**
- restaurar una copia antigua completa, CSV + sello (rollback): **rechazada: alguien ha restaurado una BD anterior (rollback)**
- borrar la BD entera, CSV + sello, para reinscribir a la victima: **rechazada: alguien ha borrado la BD**

El sistema falla cerrado: con la BD manipulada no arranca. Borrar filas sigue siendo una denegacion de servicio, pero ya no pasa desapercibido.

### 6.6 Entradas malformadas

- template NaN: rechazado: ValueError: El template contiene NaN/Inf
- template longitud 128: rechazado: ValueError: El template debe tener 512 valores (tiene 128)
- template vector nulo: rechazado: ValueError: El template es (casi) nulo
- template texto: rechazado: ValueError: could not convert string to float: 'hola'
- imagen float en [0,1]: ('adam_pearson', 1.0)
- imagen 10x10: rechazada: ValueError: Imagen demasiado pequena: (10, 10)

### 6.7 Enumeracion de identidades

verificar_t con un id inexistente devuelve (False, 0.0); con un id inscrito y un impostor devuelve (False, 0.0). Antes el primero devolvia score=None.

### 6.8 Bloqueo por intentos fallidos

5 intentos con un impostor y despues el template genuino de adam_pearson: [False, False, False, False, False, False]. Tras el 5o fallo, incluso el genuino se rechaza durante 5 minutos.

### 6.9 Fotos con varias caras (las gemelas juntas)

- commons_2018.jpg (2 caras): rechazada (ambigua)
- commons_2023.jpg (11 caras): rechazada (ambigua)
- neo2_1.jpg (2 caras): rechazada (ambigua)
- neo2_2.jpg (2 caras): rechazada (ambigua)

### 6.10 Abusar de la inscripcion

- inscribir la cara de una victima con un id nuevo (1 foto): rechazado: ValueError: Esa cara se parece mucho a una persona ya inscrita (gemelo o doble): para inscribirla hacen falta al menos 3 fotos distintas suyas
- inscribirse con 3 fotos de keanu_reeves (espejo, desenfoque, JPEG) con un id nuevo: rechazado: ValueError: Esa cara no se distingue de una persona ya inscrita ni con estas fotos: aporta fotos mas claras o mas variadas
- inscribir con 3 fotos de personas distintas: rechazado: ValueError: Las fotos de la inscripcion no parecen de la misma persona
- reemplazar la identidad de la victima sin clave de admin: rechazado: PermissionError: Modificar una identidad inscrita (reemplazar o anadir fotos) requiere la clave de administrador (define SB_CLAVE_ADMIN y pasa clave_admin / --admin)
- anadir una foto propia a la identidad de la victima sin clave de admin: rechazado: PermissionError: Modificar una identidad inscrita (reemplazar o anadir fotos) requiere la clave de administrador (define SB_CLAVE_ADMIN y pasa clave_admin / --admin)

Flujo legitimo de un gemelo (Paula, con Aitana ya inscrita):

- con 1 foto: **rechazada: Esa cara se parece mucho a una persona ya inscrita (gemelo o doble): para inscribirla hacen falta al menos 3 fotos distintas suyas**
- con sus 3 fotos: **inscrita**

![distribuciones](distribuciones.png)
