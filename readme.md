# HPCA aplicado al Merval

Trabajo práctico final de Finanzas Cuantitativas. Replicamos la metodología de **Hierarchical PCA** (Avellaneda, 2019, *Hierarchical PCA and Applications to Portfolio Management*, arXiv:1910.02310) sobre el mercado accionario argentino, para separar el riesgo sistemático de mercado del riesgo específico de cada sector.

## Contexto

El PCA tradicional sobre la matriz de correlación de retornos identifica factores comunes de riesgo (*eigenportfolios*). El primero suele interpretarse como "el mercado", pero los siguientes son difíciles de interpretar y cambian mucho de una ventana temporal a otra. El paper llama a esto el **problema de identificación**.

El HPCA lo resuelve usando información económica que ya tenemos: la partición del mercado en sectores.

1. Se hace un PCA **dentro de cada sector** y se toma su primer eigenportfolio $F^{(1,k)}$ (el "factor sector").
2. Se supone que los residuos de acciones de **sectores distintos** no están correlacionados (supuesto HPCA, ec. 10).
3. La matriz de correlación se reconstruye así:
   - dentro de un mismo sector: la correlación empírica $R_{ij}$;
   - entre sectores distintos: $\beta_i \beta_j \bar\rho^{I(i)I(j)}$, donde $\bar\rho$ es la correlación entre los factores sectoriales (ec. 11).
4. El espectro de la matriz resultante $\tilde R$ se puede calcular de forma explícita (Proposición 2):
   - los autovalores de la matriz $M_{kk'} = \sqrt{\lambda^{(1,k)}\lambda^{(1,k')}}\,\bar\rho^{kk'}$ dan los factores **multi-sector**, es decir, combinaciones de factores sectoriales;
   - los autovalores $\lambda^{(j,k)}$ con $j \ge 2$ dan los factores **long-short intra-sector**.

Así, cada factor de riesgo tiene una lectura económica directa: el mercado, un sector contra otro, o una posición long-short dentro de un sector.

## Ideas del trabajo

### Idea 1: recrear el paper con el Merval (hecha)

Aplicamos el HPCA a 27 acciones argentinas del panel líder y del panel general, agrupadas en 5 sectores: Financiero, Petróleo y Gas, Utilities, Materiales y Consumo y Otros. Usamos retornos diarios en USD-CCL entre 2018 y 2026. Todo está en `HPCA_Merval.py` y los gráficos quedan en `figuras/`.

Resultados principales:

- **El primer factor es el mercado, y HPCA explica lo mismo que PCA.**
  - EV1 explica 60,9% de la varianza con PCA y 60,7% con HPCA, con una similitud coseno de 0,999 entre ambos autovectores. En el S&P 500 del paper explicaba 30%: en el Merval domina el riesgo país.
  - Las correlaciones entre sectores son muy altas, de 0,72 a 0,87. Materiales, que son empresas exportadoras, es el sector menos correlacionado.
- **Los factores HPCA se pueden interpretar.** Por ejemplo, el EV2 es Materiales contra Financiero y el EV4 es Utilities contra Financiero. Los autovectores de PCA mezclan sectores sin una lógica clara.
- **Los residuos de los dos modelos frente a Marchenko-Pastur son prácticamente iguales.** La matriz completa no aporta más información que el HPCA, igual que en el paper.
  - Límite del análisis: solo 1 autovalor supera la cota de ruido. Con 27 acciones, los factores que siguen al mercado son estadísticamente débiles.
- **Estabilidad en el tiempo.** Los factores HPCA son más estables que los de PCA en EV2 a EV4. La salvedad es que cambian de orden entre ventanas porque sus autovalores están muy juntos.
- **Cartera de mínima varianza *out-of-sample*.** HPCA logra una volatilidad similar a la matriz muestral (37,9% contra 37,8% anual) pero con bastante menos rotación (0,50 contra 0,74 por mes).

Conclusión: en el Merval, HPCA no es más preciso que PCA, pero es igual de explicativo, mucho más interpretable y genera carteras con menos rotación.

### Idea 2: otro mercado desarrollado (revisión)

Repetir el análisis en un mercado más grande y con sectores más diversificados, por ejemplo:

- **Reino Unido:** FTSE 100 o FTSE 350, tickers `.L` en yfinance.
- **Japón:** Nikkei 225 o TOPIX, tickers `.T` en yfinance.

Motivación:

- Con 100 a 225 acciones, el análisis se acerca a las condiciones del paper (n = 434). Debería haber más factores por encima de la cota de Marchenko-Pastur, así que el aporte del HPCA para separar señal de ruido se vuelve más relevante.
- Permite comparar un mercado emergente con un mercado desarrollado.

Para adaptar el script alcanza con cambiar:

- los tickers y la partición sectorial (por ejemplo, sectores GICS o ICB);
- la moneda: usar retornos en moneda local o en USD, y eliminar el cálculo del CCL;
- el filtro de liquidez, si hiciera falta.

#### Mercado de Japón: Nikkei 255

Aplicamos el HPCA a las 225 acciones del índice Nikkei 225 (217 tras el filtro de liquidez), agrupadas en 25 sectores a partir de la industria propia de Nikkei. Usamos retornos diarios en JPY entre 2018 y 2026, sin ajuste cambiario porque Japón no tiene control de cambios. Todo está en `HPCA_Nikkei.py` y los gráficos quedan en `figuras/nikkei/`.

Resultados principales:

- **El primer factor es el mercado, y HPCA explica casi lo mismo que PCA.**
  - EV1 explica 36,3% de la varianza con PCA y 36,0% con HPCA, con una similitud coseno de 0,995. Queda entre el 30% del S&P 500 del paper y el 60,9% del Merval: Japón es un mercado grande y diversificado, pero sigue siendo un único país.
  - Las correlaciones entre sectores van de 0,30 a 0,88 (promedio 0,58), bastante más dispersas que en el Merval (0,72–0,87). Marítimo es el sector menos correlacionado con el resto (el transporte naval depende más del ciclo de comercio global que del mercado japonés), mientras que el cluster industrial-exportador (Maquinaria Eléctrica, Materiales, Maquinaria Industrial, Vidrio y Cerámica) es el más correlacionado entre sí.
- **Los factores HPCA se pueden interpretar.** El EV2 es Alimentos y Ferrocarriles contra Maquinaria Eléctrica; el EV3 es Banca contra Maquinaria Eléctrica y Tecnología; el EV4 es un corto puro en Tecnología, Banca y Seguros. Los autovectores de PCA vuelven a mezclar sectores sin una lógica clara.
- **Con 217 acciones, el análisis se acerca mucho más a las condiciones del paper que el Merval.** Hay 11 autovalores por encima de la cota de Marchenko-Pastur (λ+ = 1,74) en la muestra completa, contra 1 solo en el Merval. Al remover los 25 factores sectoriales, los residuos de PCA y HPCA quedan prácticamente iguales (27 autovalores por encima de la cota en ambos casos): la matriz completa no aporta más información que el HPCA, igual que en el paper.
- **Estabilidad en el tiempo: resultado más parejo que en el Merval.** A nivel de autovector individual, PCA y HPCA quedan muy parecidos (por ejemplo EV3: 0,945 PCA vs 0,956 HPCA; EV4: 0,917 vs 0,907), sin una ventaja sistemática de uno sobre otro. A nivel de subespacio, menos sensible a cambios de orden entre autovalores cercanos, HPCA es levemente más estable (top-3: 0,984 contra 0,972; top-5: 0,956 contra 0,948).
- **Cartera de mínima varianza *out-of-sample*: acá sí hay una ventaja clara de HPCA.** Logra el mejor retorno ajustado por riesgo (Ret/Vol 0,545, contra 0,398 de PCA y 0,520 de Ledoit-Wolf) con una rotación mensual muy inferior a ambos (0,74 contra 1,43 de PCA y 2,88 de Ledoit-Wolf). La matriz muestral sin estructura es inusable en la práctica: necesita 14,9 de rotación mensual y un apalancamiento bruto de 16,6 para lograr, de todos modos, una volatilidad mucho más alta (28,6% anual).
- **Robustez JPY vs USD: la varianza explicada no cambia, pero el signo de la relación con el tipo de cambio sí.** El EV1 explica prácticamente lo mismo en yenes que en dólares (36,0% vs 35,9% con HPCA), pero su correlación con el USD/JPY cambia de signo: +0,23 en yenes (las acciones japonesas suben cuando el yen se devalúa, el clásico efecto exportador) y −0,21 en dólares (ese mismo movimiento, traducido a dólares, se revierte). Confirma que medir en yenes aísla mejor la estructura accionaria del ruido cambiario.
- **Robustez sectorial: 25 sectores (industria Nikkei) vs 17 sectores (JPX oficial) dan prácticamente el mismo resultado.** EV1 HPCA explica 36,0% con la partición propia y 35,9% con la oficial de JPX; el error de aproximación fuera de bloque es 0,056 y 0,060 respectivamente. El resultado no depende de qué tan fina sea la clasificación sectorial, siempre que sea económicamente razonable.

Conclusión: a diferencia del Merval, en el Nikkei 225 HPCA no sólo es igual de explicativo e interpretable, sino que además arma carteras de mínima varianza con mejor retorno ajustado por riesgo y mucha menos rotación, probablemente porque con 217 acciones y 25 sectores hay bastante más margen real para que la estructura jerárquica le gane al ruido estadístico.

## Objetivo

- Construir la matriz HPCA para acciones argentinas agrupadas por sector y compararla con el PCA tradicional en:
  - varianza explicada;
  - forma de los primeros autovectores;
  - residuos frente a la cota de Marchenko-Pastur.
- Evaluar si el HPCA es **igual de explicativo y más interpretable y estable** que el PCA en un mercado chico, concentrado y con fuertes shocks macro.
- Discutir las aplicaciones a gestión de carteras: proxy del portafolio de mercado, cobertura sectorial y construcción de carteras de mínima varianza.

## Particularidades del mercado argentino

A diferencia del S&P 500 del paper (434 acciones), el universo local es chico: entre 20 y 60 acciones según se use el Merval o el panel general. Además tiene rasgos que afectan las correlaciones:

- **Precios en pesos con alta inflación y devaluaciones discretas.** Un factor "peso/CCL" común puede dominar el primer autovector. Conviene comparar los resultados con precios en ARS y en USD (CCL).
- **Baja liquidez en parte del panel.** Los precios que no se actualizan (retornos iguales a cero) sesgan las correlaciones hacia abajo.
- **Shocks de régimen.** Por ejemplo, las PASO de agosto de 2019 o las elecciones de 2023. Pocos días extremos pueden dominar la matriz de correlación.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `HPCA_Merval.py` | Script principal en formato notebook (celdas `# %%`). |
| `pyproject.toml` | Dependencias del proyecto, gestionadas con `uv`. |
| `uv.lock` | Versiones exactas de las dependencias. Garantiza que todo el grupo tenga el mismo entorno. |
| `figuras/` | Gráficos que genera el script, en PNG, para usar en el informe. |
| `readme.md` | Este archivo. |

Secciones de `HPCA_Merval.py`:

1. Configuración: fechas, moneda (ARS o USD-CCL), partición sectorial y filtros.
2. Descarga de precios con `yfinance` y cálculo del dólar CCL implícito (GGAL y YPF).
3. Limpieza: filtro de iliquidez, winsorización y estandarización.
4. Implementación de PCA y HPCA, con verificación numérica de las Proposiciones 1 y 2 del paper.
5. Varianza explicada e interpretación de cada eigenportfolio (equivalente a la Tabla 2 del paper).
6. Comparación de los primeros 5 autovectores PCA vs HPCA.
7. Matrices de correlación empírica, HPCA y la diferencia entre ambas.
8. Residuos vs Marchenko-Pastur.
9. Estabilidad temporal de los autovectores (ventanas rolling).
10. Carteras de mínima varianza *out-of-sample*: muestral, PCA, HPCA, Ledoit-Wolf y equiponderada.
11. Robustez: ARS vs USD y partición de 3 vs 5 sectores.

## Cómo correrlo

### 1. Instalar `uv`

[`uv`](https://docs.astral.sh/uv/) es un gestor de entornos y paquetes de Python. Reemplaza a `pip` + `venv` y es mucho más rápido. Se instala una sola vez por computadora:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# o con Homebrew
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Para verificar la instalación: `uv --version`.

### 2. Crear el entorno

Desde la carpeta del proyecto:

```bash
uv sync
```

Este comando:

- crea el entorno virtual en `.venv/`;
- descarga la versión de Python que haga falta, si no está instalada;
- instala exactamente las versiones fijadas en `uv.lock`.

No hace falta activar el entorno a mano.

Si en algún momento hay que agregar una librería, se usa `uv add nombre-libreria`. Ese comando actualiza `pyproject.toml` y `uv.lock`; los dos archivos se commitean para que el resto del grupo quede sincronizado con un `uv sync`.

### 3a. Correr en VS Code, celda por celda (recomendado)

`HPCA_Merval.py` es un **script `.py` con formato de notebook**. Las líneas `# %%` separan celdas, igual que en un Jupyter `.ipynb`. Pasos:

1. Instalar las extensiones **Python** y **Jupyter** de Microsoft en VS Code.
2. Abrir la carpeta del proyecto y elegir el intérprete del entorno: `Ctrl/Cmd + Shift + P` → *Python: Select Interpreter* → `.venv`.
3. Arriba de cada `# %%` aparece **Run Cell**. También se puede usar `Shift + Enter`. Cada celda se ejecuta en la *Interactive Window*, con los gráficos y las tablas en línea, como en un notebook. Las variables quedan en memoria, así que se puede volver a correr una sola celda sin repetir la descarga de datos.

Ventajas de este formato frente a un `.ipynb`:

- **Mejor para trabajar con IA** (Claude Code, Copilot, Cursor, etc.). Es texto plano, sin el JSON, las salidas ni las imágenes en base64 que guarda un `.ipynb`. El asistente lo lee y lo edita entero, con menos tokens y menos errores, y sus cambios se pueden revisar línea por línea.
- **Diffs limpios en git.** Dos personas pueden editar el archivo y resolver conflictos como en cualquier código. Con un `.ipynb`, cada ejecución cambia el archivo aunque el código sea el mismo.
- **Es un script normal.** Se puede correr completo desde la terminal o importar funciones (`hpca`, `verificar_hpca`) desde otro archivo.

### 3b. Correr el script completo desde la terminal

```bash
uv run python HPCA_Merval.py
```

`uv run` usa automáticamente el entorno del proyecto. Los gráficos se guardan en `figuras/` y las tablas se imprimen en la consola.

### Parámetros principales

Están en la primera celda del script:

| Parámetro | Default | Qué controla |
|---|---|---|
| `MONEDA` | `"USD"` | `"USD"` convierte los precios por CCL; `"ARS"` usa los precios en pesos. |
| `SECTORES` | `SECTORES_5` | Partición de 5 sectores, o `SECTORES_3` (la original de 3). |
| `INICIO`, `FIN` | 2018-01 a 2026-09 | Período de análisis. |
| `VENTANA`, `PASO` | 252, 21 | Días de estimación y de rebalanceo en los análisis *rolling*. |
| `MAX_FRAC_CEROS` | 0.10 | Máxima fracción de días sin variación de precio (filtro de liquidez). |
| `WINSOR_SIGMA` | 5 | Recorte de retornos extremos, en desvíos estándar. |

> **Nota sobre los datos:** `yfinance` no tiene datos de algunos tickers locales (por ejemplo, `VIST.BA`). Además, a veces publica precios erróneos o desactualizados. El filtro de liquidez y la winsorización mitigan el problema, pero conviene revisar las acciones que el script descarta e informarlo.

## Referencias

- Avellaneda, M. (2019). *Hierarchical PCA and Applications to Portfolio Management*. arXiv:1910.02310.
- Avellaneda, M. y Lee, J.H. (2010). *Statistical arbitrage in the US equities market*. Quantitative Finance, 10(7).
- Laloux, L., Cizeau, P., Potters, M. y Bouchaud, J.-P. (2000). *Random matrix theory and financial correlations*.
