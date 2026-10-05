# MARCO_KAN — marco experimental KAN × FitSNAP

Reescritura del andamiaje que se rompió el 27–28 de septiembre, más los arreglos
y mediciones de la sesión del 1 de octubre.

## Los archivos

El notebook ejecutable ya no vive aquí: está en `04_CODIGO/notebooks/`, porque
viaja al servidor en la raíz del paquete y no dentro de esta carpeta.

| Archivo | Qué es |
|---|---|
| `kan_lib.py` | El módulo. Capas KAN, los siete parches de compatibilidad, `ajustes_fitsnap`, `cambiar_arquitectura`, `una_corrida`, las medidas de influencia, las 8 condiciones |
| `barrida.py` | Calibración de lote (`--que lote`), de tasa (`--que lr`) y de modo de rango (`--que modo`) |
| `prueba_variantes.py` | Compara los cuatro modos de rango sin GPU ni FitSNAP. Segundos |
| `trabajador.py` | Un proceso, un lote de corridas |
| `lanzar.py` | Reparte las corridas entre las tarjetas y recuenta al final |
| `monitor_fases.py` | Mide recursos separando descriptores de entrenamiento |
| `comparar.py` | Envoltorio de `compare_runs.py`: tabla contra baseline y parity plots |
| `compare_runs.py` | **Copia literal** del script de `fitsnap-custom` (Arbués). Ver procedencia |
| `verificar_lib.py` | Lista nombres usados y no definidos. Un segundo |
| `verificar_orden.py` | Dice si un notebook corre de arriba abajo |
| `sonda_gradientes.py` | Mide el gradiente que llega a cada capa dentro del pipeline real |
| `mae.py` | **El criterio.** Errores físicos desde las predicciones, con el detector de «predice cero» |
| `tabla70.py` | Junta las corridas largas y marca sola las que divergieron o cayeron en el suelo |
| `controles70.sh`, `estables70.sh`, `referencia70.sh` | Las tandas de la campaña del 2 de octubre, conservadas para reproducirlas |
| `migraciones/` | Parches y generadores de una sola vez, ya ejecutados. Se conservan como registro de cómo llegó el código a su estado actual |

## Orden de uso en el servidor

```bash
cd /workspace/kan_snap/MARCO_KAN
export PY=/venv/fitsnap/bin/python     # el prompt dice (main), que NO es FitSNAP
```

```bash
# 1. Comprobación estática. No gasta GPU, tarda un segundo.
$PY lanzar.py --solo-comprobar

# 2. Que el diagnóstico aplique a la versión instalada.
$PY -c "
import inspect
from kan_lib import aplicar_todos_los_parches, NORMALIZAR_KAN
aplicar_todos_los_parches()
import fitsnap3lib.solvers.pytorch as P
import fitsnap3lib.io.sections.solver_sections.pytorch as S
print('model_best:', 'self.model = self.model_best' in inspect.getsource(P))
print('dtype_setting:', 'dtype_setting' in inspect.getsource(S))
print('normalizacion KAN:', NORMALIZAR_KAN)"

# 2b. Los cuatro modos de rango, sin GPU ni FitSNAP. Segundos.
$PY prueba_variantes.py

# 3. Las 8 condiciones, una semilla, pocas épocas.
KAN_LR=5e-4 $PY lanzar.py --raiz /workspace/kan_snap --semillas 1111 --epocas 3 --si

# 4. La comparación contra el baseline.
$PY comparar.py /workspace/kan_snap/corrida_AAAAMMDD_HHMM --system --plots

# 5. Calibración, el lote primero.
$PY barrida.py --que lote --raiz /workspace/kan_snap
$PY barrida.py --que lr   --raiz /workspace/kan_snap

# 5b. Estudio lateral: los cuatro modos de rango, un solo cálculo de descriptores.
$PY barrida.py --que modo --raiz /workspace/kan_snap \
   --arquitectura gaussiana --semillas 1111 2222 --epocas 3

# 6. La rejilla, solo cuando 3 y 4 salgan bien.
$PY lanzar.py --raiz /workspace/kan_snap --epocas 70
```

## La disposición de salida

`trabajador.py` escribe siguiendo la convención de `workspace/07_experimentos/`
de fitsnap-custom, para que `compare_runs.py` lea estas corridas sin cambios:

```
<salida>/<grupo>/run_NN_<id>_s<semilla>/
    <id>_s<semilla>_metrics.dat     <- renombrado; FitSNAP lo llama metrics.dat
    loss_vs_epochs.dat
    perconfig.dat                   <- de aquí salen los parity plots
    peratom.dat
    system_metrics.csv              <- recursos de ESTA corrida
    pot.mod, pot.mliap.descriptor, FitTorch_Pytorch.pt, modelo.pt
```

El `NN` sale del prefijo numérico de la condición, así que `1_mlp_base` cae en
`run_01_...` y `compare_runs.py` lo toma como baseline automáticamente.

**Por qué hay que mover archivos.** FitSNAP resuelve `metrics.dat`,
`peratom.dat`, `perconfig.dat` y los modelos al **construir** el objeto, no al
escribir. Todas las corridas de un mismo trabajador escriben en la carpeta del
trabajador y se pisan entre sí: el 1 de octubre solo sobrevivió el `metrics.dat`
de la última corrida de cada uno. El trabajador los mueve por fecha de
modificación, sin depender de internos de FitSNAP.

## Las comprobaciones incorporadas

**Nombres sin definir, antes de lanzar.** `verificar_lib.py` recorre el árbol
sintáctico. `PESOS`, `pesos_de` y `_ORIGINALES` fueron el mismo fallo tres veces,
a 8–15 minutos de GPU cada uno. Limitación conocida: detecta globales que faltan,
**no métodos** que faltan — se invocan como atributo y el análisis no los ve.

**Optimizador acoplado, antes de cada ajuste.** Compara por identidad de objeto
los parámetros del modelo y los del optimizador. Habría cortado en seco la
sesión del 28 de septiembre.

**Los pesos se movieron, después de cada ajuste.** Una corrida con `delta = 0` se
marca como no entrenada aunque haya escrito `metrics.dat`. Limitación: `delta`
está dominado por la capa de estandarización, que `perform_fit` sobreescribe, así
que distingue el caso roto pero no mide cuánto se aprendió.

**Y la que falta en el código: mirar el parity plot.** Un modelo congelado predice
casi una constante y sale como línea horizontal. El 1 de octubre eso se descubrió
analizando ocho archivos a mano; con `comparar.py --plots` se ve de un vistazo.

## Variables de entorno

| | |
|---|---|
| `KAN_NORMALIZAR` | cómo se gestiona el rango de entrada del bloque KAN: **`no` (defecto)**, `canal`, `layernorm` o `cuantiles`. Se aceptan los valores viejos: `1` → `canal`, `0` → `no`. El defecto cambió de `canal` a `no` el 3 de octubre: congelar estadísticas resultó ser sesenta veces peor que no normalizar |
| `KAN_RESIDUO` | `1` añade el término residual de Liu et al., `w_b·silu(x)` en paralelo a la base. Cuesta `n_in × n_out` por capa KAN: 128 parámetros en la condición principal. Apagado por omisión |
| `KAN_REJILLA_CADA` | cada cuántos pasos de entrenamiento se recalcula la rejilla a los cuantiles de los datos. `0` (defecto) = nunca |
| `KAN_LR`, `KAN_BATCH_SIZE`, `KAN_FRACCION` | pisan los valores de `kan_lib.py` |
| `KAN_DTYPE_SETTING` | sin efecto: la clave no está en la lista blanca de FitSNAP y el envoltorio ya no la envía. Si se fija, se avisa y se ignora |
| `KAN_RAIZ` | carpeta de trabajo |

## Lo que cambió, y por qué

**La normalización de entrada del bloque KAN** (1 de octubre). Medido: con la
tasa corregida los tres MLP caen 96 % en 3 épocas y los cinco KAN entre 0.3 y
2.0 %. La entrada al bloque de salida tiene media 0.72 y desviación 0.22 —el
`Softplus` previo solo da positivos— contra bases centradas en (-2, 2): tres de
las ocho campanas gaussianas reciben activación media menor que 0.01, y el
gradiente de la peor servida es 2.1e-05.

La capa KAN estandariza ahora su propia entrada con estadísticas del primer lote,
congeladas después y guardadas como buffers. Resultado: 0 campanas muertas, y el
gradiente de la peor pasa a 34.4.

No favorece al KAN. Una capa `Linear` absorbe la escala de su entrada en sus
pesos; una base fija no puede. Corrige un sesgo que iba en contra.

**Tres formas de hacerlo, no una.** Llevar los datos a la rejilla con media y
desviación y mover la rejilla a los datos con esa misma media y desviación son
el mismo modelo reparametrizado —comprobado numéricamente, diferencia 3.6e-07—,
así que no son dos rutas. Lo que sí son distintas rutas:

| modo | qué hace | de dónde viene |
|---|---|---|
| `canal` | estandariza por canal, congelado tras el primer lote. Rejilla uniforme | el defecto |
| `layernorm` | normaliza por muestra a lo ancho de los canales, en cada paso | FastKAN |
| `cuantiles` | mueve la rejilla a los cuantiles de los datos. Rejilla **no** uniforme | pykan, `update_grid_from_samples` |

Gradiente que recibe la componente peor servida, medido con la distribución real
de entrada al bloque de salida:

| modo | gaussiana | bspline | chebyshev |
|---|---|---|---|
| `no` | 1.4e-09 | **0.0e+00** | 2.3e-01 |
| `canal` | 2.7e-02 | 1.5e-03 | 2.0e-01 |
| `layernorm` | 2.7e-02 | 2.9e-03 | 1.7e-01 |
| `cuantiles` | 5.5e-02 | 1.7e-03 | no aplica |

Dos lecturas. La B-spline sin corregir recibe gradiente **exactamente** cero: su
soporte es compacto, no asintótico, así que una entrada fuera del último nudo no
deja ninguna derivada. Y el Chebyshev no mejora al corregirlo, porque su `tanh`
previo ya gestionaba el rango. La respuesta no es la misma para las tres bases, y
por eso conviene medirla. `cuantiles` no aplica al Chebyshev: su base es global,
no tiene rejilla.

Los cuatro modos tienen el mismo número de parámetros, así que la comparación no
mezcla rango con capacidad. Se reporta como nota de método, no como factor de la
rejilla: la rejilla corre con `canal`.

**El módulo ya no se genera en cada corrida.** `escribir_libreria` volcaba
funciones con `inspect.getsource` desde el kernel vivo y no arrastraba los
globales de los que dependían.

**El monitor separa fases.** El anterior promediaba la utilización de GPU sobre
toda la corrida; como 452 de cada 456 segundos son descriptores en CPU, el
promedio salía entre 0 y 9 %.

## Procedencia

`compare_runs.py` es una **copia literal** (sin modificar, md5 verificado) de
`workspace/compare_runs.py` del repositorio `fitsnap-custom` de Arbués, cuyo uso
ofreció explícitamente en la reunión del 27 de septiembre. `comparar.py` es un
envoltorio propio que apunta su `07_experimentos/` a la carpeta de corrida
indicada, precisamente para no tocar su archivo y poder reemplazarlo por una
versión suya más nueva.

`monitor_fases.py` sigue la arquitectura de su `monitor_training.py` y conserva
los nombres de columna de su CSV.

No se toma nada más. El `disk_cache` sobre HDF5 y el corte de seguridad por RAM
al 90 % son intervenciones en el pipeline y quedan fuera por el filtro de
independencia del diseño. Un instrumento de medición no entra en ese filtro,
pero conviene dejar constancia en el manuscrito.

## Si algo falla

| Síntoma | Dónde mirar |
|---|---|
| `ModuleNotFoundError: pandas` | Es el intérprete. `export PY=/venv/fitsnap/bin/python` |
| `unmatched variable: dtype_setting` | El envoltorio la escribía siempre, con 1 por defecto; el arreglo del 1 de octubre solo tocó `trabajador.py` y quedó incompleto. Desde el 2 de octubre no se escribe nunca. float64 sigue sin vía |
| Un trabajador muere y el log solo trae el pie del monitor | `--silencioso` se tragaba el traceback. Quitado de `lanzar.py` |
| `verificar_lib.py` lista nombres que faltan | Añadirlos a `kan_lib.py` |
| "optimizador desacoplado antes del ajuste" | Correr una condición por proceso |
| Todas las corridas con `delta = 0` | Revisar primero la comprobación 2 de arriba |
| El CSV sale con fase "Desconocida" | Los textos de `TRANSICIONES` no coinciden con esta versión de FitSNAP |
| `comparar.py` no encuentra corridas | La salida no sigue la convención. Comprobar que hay `*_metrics.dat` dentro de `<grupo>/run_NN_.../` |

## Advertencia

Lo de la sesión del 1 de octubre está probado en servidor. Lo añadido después
—normalización del KAN y sus tres variantes, rutas de salida, la disposición
compatible con `compare_runs.py`— está probado offline: sintaxis, comprobación
estática, el módulo real importado en los cuatro modos con las tres bases
(gradientes, forma de salida, float64, buffers y calibración congelada en
`eval`) y un árbol de prueba para la comparación. En servidor, sin correr
todavía.
