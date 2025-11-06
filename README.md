# TP5 - Simulación de Sistemas

El presente trabajo implementa una dinámica 2D de peatones autopropulsados con condiciones periódicas, un obstáculo central fijo y reglas de evasión anisotrópicas, replicando el modelo AA-CPM. Cada partícula alterna entre expansión/contracción de su radio efectivo y estrategias locales de evasión para evitar solapes, permitiendo estudiar el régimen estacionario de contactos con el obstáculo.

Las funcionalidades incluidas son las siguientes:

- <b>Motor de peatones en 2D</b>: Evoluciona partículas móviles con radios adaptativos, velocidades deseadas y regla de evitación basada en vecinos cercanos, incluyendo una partícula fija central que registra contactos.
- <b>Escenarios configurables</b>: Permite barrer distintos tamaños de población (`N`), pasos temporales (`dt`), duración (`t_f`) y granularidad de guardado (`step`) vía propiedades del sistema o scripts de batch.
- <b>Persistencia de simulaciones</b>: Guarda cada corrida en `data/simulations/<nombre>/` con `static.txt` (atributos globales y estados F/M) y `dynamic.txt` (trazas temporales con hits acumulados por frame).
- <b>Métricas de contactos</b>: El motor registra los cruces de la frontera periódica, los contactos con el obstáculo y el conteo acumulado `N_c(t)` para analizar régimen estacionario y ruido entre réplicas.
- <b>Post-procesamiento experto</b>: Scripts en Python para promediar réplicas, ajustar pendientes por mínimos cuadrados, estimar barras de error y construir el gráfico global `Q` vs `ϕ` junto con distribuciones `P(τ ≥ τ_min)`.
- <b>Visualización</b>: Animaciones 2D en Matplotlib que muestran radios instantáneos, vectores velocidad y resaltan partículas en contacto con el centro, además de curvas temporales con zoom configurable.
- <b>Informe técnico</b>: Carpeta `report/` con el documento del trabajo práctico y los recursos gráficos generados automáticamente.

<details>
  <summary>Contenidos</summary>
  <ol>
    <li><a href="#instalación">Instalación</a></li>
    <li><a href="#instrucciones">Instrucciones</a></li>
    <li><a href="#manual-de-usuario">Manual de Usuario</a></li>
    <li><a href="#integrantes">Integrantes</a></li>
  </ol>
</details>

## Instalación

Clonar el repositorio:

- HTTPS:
  ```sh
  git clone https://github.com/martinAleB/sds-tp5.git
  ```
- SSH:
  ```sh
  git clone git@github.com:martinAleB/sds-tp5.git
  ```

Motor de simulación (Java + Maven):

```sh
cd sds-tp5/simulations
mvn clean package
```

Scripts de post-procesamiento (Python >= 3.10):

```sh
cd sds-tp5
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install numpy matplotlib powerlaw
```

> **Requisitos**: JDK 21 (compatible con Maven), Python 3.10+ con `pip`, y FFmpeg para exportar animaciones MP4.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

## Instrucciones

Todos los comandos deben ejecutarse desde la raíz del repositorio (o la carpeta indicada) con el entorno correspondiente activado.
En los ejemplos, `baseA`, `baseB` y `baseB_1` representan nombres definidos por el usuario al lanzar las simulaciones (sus réplicas siguen el patrón `<base>_1`, `<base>_2`, `<base>_3`).

- Compilación rápida del motor:
  ```sh
  cd simulations
  mvn clean package
  ```
- Simulación individual:
  ```sh
  java -cp simulations/target/classes \
    -DN=<N> -Ddt=<dt> -Dt_f=<t_f> -Dstep=<step> \
    -Dname=<nombre> \
    ar.edu.itba.sds.tp5.simulations.Engine
  ```
- Lote de realizaciones (3 réplicas por N):
  ```sh
  cd simulations
  ./run_batch.sh
  ```
- Gráfico de hits promedio con zoom:
  ```sh
  python postprocessing/plot_hits.py baseA baseB --tmark <tmark>
  ```
- Barrido de pendientes y barras de error:
  ```sh
  python postprocessing/contacts_linear_fit.py baseA baseB --tmark <tmark>
  ```
- Curva global Q vs ϕ:
  ```sh
  python postprocessing/scanning_rate.py baseA baseB --tmark <tmark>
  ```
- Ajuste power-law de tiempos entre contactos:
  ```sh
  python postprocessing/alpha_vs_phi.py --t0 <t0> baseA baseB
  ```
- Animación 2D de una corrida:
  ```sh
  python postprocessing/animate.py baseB_1 --slow <factor>
  ```

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

## Manual de Usuario

### Motor de peatones (Java)

```sh
java -cp simulations/target/classes \
  -DN=<N> \
  -Ddt=<dt> \
  -Dt_f=<t_f> \
  -Dstep=<step> \
  -Dname=<nombre> \
  ar.edu.itba.sds.tp5.simulations.Engine
```

Parámetros:

- `name`: carpeta de salida en `data/simulations/<name>/`.
- `N`: cantidad de peatones móviles (se suma el disco central fijo).
- `dt`: paso temporal del integrador (segundos).
- `t_f`: tiempo total a simular.
- `step`: frecuencia con la que se vuelca un frame a `dynamic.txt`.

Archivos generados:

- `static.txt`: constantes globales (`L`, `N`, `r_min`, `r_max`) y el listado de estados `F/M`.
- `dynamic.txt`: por cada snapshot, una línea con `t` y `N_c(t)` seguida por `N+1` filas con `(x, y, vx, vy, r, touchedCentral)`.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Lote de simulaciones

```sh
cd simulations
./run_batch.sh
```

- Ejecuta las combinaciones predefinidas de `N` (ver array `NS` en el script) con `t_f=600` y tres réplicas.
- Cada corrida queda (por defecto) en `data/simulations/test<N>_600_<rep>/`, pero se puede editar `run_batch.sh` para definir un prefijo distinto.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Curvas de hits promedio

```sh
  python postprocessing/plot_hits.py baseA baseB --tmark <tmark>
  python postprocessing/plot_hits_no_zoom.py baseA baseB --tmark <tmark>
```

- `plot_hits.py` promedia las tres réplicas `<base>_1.._3`, grafica `N_c(t)` y agrega un recuadro con zoom temporal.
- `plot_hits_no_zoom.py` replica la curva sin inset para figuras más simples.
- `--tmark` fija la línea vertical que separa la fase transitoria y la estacionaria (usar el valor deseado en `<tmark>`).

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Ajustes lineales y error vs pendiente

```sh
python postprocessing/contacts_linear_fit.py baseA baseB \
  --tmark <tmark> --t0 <t0> --out data/graphics/hits
```

- Calcula `N_c(t)` promedio por base, recorta `t ≥ tmark`, escanea la pendiente `b` y marca el mínimo de SSE.
- Genera dos gráficos: `error_vs_q.png` (SSE vs `Q`) y `fit_lines.png` (ajuste lineal sobre las curvas).
- `--tmark` define el inicio del intervalo estacionario (ingresar `<tmark>` acorde al análisis).
- `--t0` agrega una línea vertical informativa en los gráficos (ingresar `<t0>`).

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Tasa de escaneo Q vs ϕ

```sh
python postprocessing/scanning_rate.py baseA baseB --tmark <tmark>
```

- Para cada base `<name>`, busca las réplicas `<name>_1.._3`, calcula `ϕ`, estima `Q` con mínimos cuadrados y combina barras de error entre réplicas.
- Produce `data/graphics/hits/Q_vs_phi.png` y reporta en consola la ruta del archivo.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Distribución de tiempos entre contactos

```sh
python postprocessing/alpha_vs_phi.py --t0 <t0> baseA baseB
```

- Lee `dynamic.txt`, extrae las marcas temporales de nuevos contactos con el centro (τ), y ajusta una distribución power-law usando `powerlaw.Fit`.
- Reporta `α`, `σ_α`, `KS`, `p-value`, `τ_min` y el tamaño de la cola para cada `ϕ`.
- Genera: `alpha_vs_phi_<timestamp>.png`, `log-log_<timestamp>.png` y `p-value_vs_phi_<timestamp>.png` en `data/graphics/`.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

### Animación 2D de la simulación

```sh
python postprocessing/animate.py baseB_1 --slow <factor>
```

- Reproduce la trayectoria de cada partícula, resalta contactos con el disco central y muestra vectores velocidad.
- `--slow` multiplica el intervalo base entre frames (proveer `<factor>` según la velocidad deseada; e.g. 1.5 ⇒ animación 50 % más lenta).
- Exporta `data/animations/<nombre>.mp4` (si FFmpeg está disponible) o un GIF de fallback.

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>

## Integrantes

Martín Alejandro Barnatán (64463) - mbarnatan@itba.edu.ar  
Ignacio Martín Maruottolo Quiroga (64611) - imaruottoloquiroga@itba.edu.ar  
Ignacio Pedemonte Berthoud (64908) - ipedemonteberthoud@itba.edu.ar

<p align="right">(<a href="#tp5---simulación-de-sistemas">Volver</a>)</p>
