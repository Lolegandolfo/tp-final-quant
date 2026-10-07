# %% [markdown]
# # HPCA aplicado al mercado británico (FTSE 100 y FTSE 250)
#
# Replicación de Avellaneda (2019), *Hierarchical PCA and Applications to Portfolio
# Management*, sobre acciones cotizadas en la London Stock Exchange.
#
# Al igual que Japón, Reino Unido no tiene control de cambios: la libra flota libremente, así
# que se trabaja directamente en GBP y el dólar se usa sólo como robustez (sección 13), para ver
# si el primer factor es "mercado británico" o está contaminado por el tipo de cambio GBP/USD.
# Esto importa especialmente en el FTSE 100, donde muchas empresas generan buena parte de sus
# ingresos en moneda extranjera.
#
# Universo: 143 acciones propuestas (aprox. 100 del FTSE 100 y el resto del FTSE 250), Se priorizaron 
# empresas con historia continua desde 2018 y ticker estable; las que no se pudieron descargar o no 
# cumplen los filtros de calidad (sección 3) se descartan automáticamente. La muestra final es de n = 141
# acciones. Los sectores (11, mínimo 6 acciones cada uno) son una agrupación propia,
# inspirada en las categorías tipo GICS/ICB pero NO una clasificación oficial: por ejemplo, la
# construcción y vivienda se separa del consumo discrecional, y la energía se une a los materiales.
# Como robustez sectorial (sección 14) se compara contra una partición gruesa de 3 sectores
# (Financiero e Inmobiliario / Cíclicos / Defensivos y Tech).
#
# Limitación: se usan los componentes actuales de los índices para mirar 2018-2026, así que hay
# sesgo de supervivencia (las empresas que quebraron o fueron adquiridas no están). Es tolerable
# para estudiar correlaciones y estabilidad de factores, pero hay que tenerlo en cuenta al leer
# los retornos de la cartera de mínima varianza (sección 12).
#
# Estructura:
# 1. Configuración y descarga de datos (yfinance, tickers con sufijo .L)
# 2. Retornos en GBP y limpieza (sin necesidad de ajuste cambiario)
# 3. Implementación de PCA y HPCA (ecs. 9-16 del paper) + verificación de la Proposición 2
# 4. Espectro, varianza explicada e interpretación de los eigenportfolios (Tabla 2 del paper)
# 5. Comparación de autovectores PCA vs HPCA (Figs. 2-13 del paper)
# 6. Residuos vs Marchenko-Pastur (Sección 6 / Fig. 14 del paper)
# 7. Estabilidad temporal de los autovectores (problema de identificación)
# 8. Cartera de mínima varianza out-of-sample (aplicación a portfolio management)
# 9. Robustez: GBP vs USD
# 10. Robustez: sectores propios (11) vs partición gruesa (3)

# %% 1. Imports y configuración
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
import time

INICIO = "2015-01-01" 
FIN = "2026-09-30"
MONEDA = "GBP"

SECTORES_UK = {
    "Financiero": ["HSBA", "LLOY", "BARC", "NWG", "STAN", "LGEN", "PRU", "AV", "PHNX", "STJ",
                   "ADM", "HSX", "BEZ", "III", "LSEG", "ICG", "SDR", "ABDN", "IGG", "PLUS",
                   "CBG", "PAG"],
    "Recursos": ["SHEL", "BP", "RIO", "AAL", "GLEN", "ANTO", "FRES", "CRDA", "JMAT", "MNDI",
                 "HOC", "VCT", "SYNT"],
    "Industriales": ["BA", "RR", "SMIN", "RTO", "IMI", "WEIR", "SPX", "BNZL", "REL", "EXPN",
                     "ITRK", "HLMA", "DPLM", "RS1", "MRO", "QQ", "AHT", "IAG", "EZJ", "DCC",
                     "ROR", "VSVS", "CHG", "SRP", "BAB", "HAS", "PAGE"],
    "Consumo básico": ["ULVR", "DGE", "BATS", "IMB", "RKT", "TSCO", "SBRY", "CCH", "ABF",
                       "TATE", "CWK", "BME"],
    "Consumo discrecional": ["NXT", "MKS", "KGF", "JD", "IHG", "CPG", "ENT", "OCDO", "DNLM",
                             "PETS", "BRBY", "WTB", "GRG", "GAW", "DOM", "JDW", "FRAS", "CARD"],
    "Construcción y vivienda": ["PSN", "TW", "BKG", "BWY", "VTY", "BBY", "MGNS", "HWDN", "TPK"],
    "Salud": ["AZN", "GSK", "SN", "HIK", "CTEC", "GNS"],
    "Tecnología y digital": ["SGE", "RSW", "OXIG", "CCC", "KNOS", "NCC", "GBG", "RMV", "AUTO"],
    "Comunicaciones y medios": ["VOD", "BT-A", "WPP", "ITV", "PSON", "INF", "FUTR"],
    "Utilities": ["NG", "SSE", "CNA", "SVT", "UU", "PNN", "DRX"],
    "Inmobiliario": ["LAND", "BLND", "SGRO", "UTG", "LMP", "GPE", "DLN", "BBOX", "HMSO",
                     "BYG", "SAFE", "WKP", "PHP"]
}
SECTORES = SECTORES_UK

SECTORES_3 = {
    "Financiero e Inmobiliario": SECTORES["Financiero"] + SECTORES["Inmobiliario"],
    "Cíclicos": (SECTORES["Recursos"] + SECTORES["Industriales"] + SECTORES["Consumo discrecional"]
                 + SECTORES["Construcción y vivienda"]),
    "Defensivos y Tech": (SECTORES["Consumo básico"] + SECTORES["Salud"] + SECTORES["Utilities"]
                          + SECTORES["Tecnología y digital"] + SECTORES["Comunicaciones y medios"])}

# Filtros de calidad de datos
MAX_FRAC_CEROS = 0.10  # descarta acciones con >10% de días sin variación (iliquidez)
WINSOR_SIGMA = 5.0     # recorta retornos diarios a ±5 desvíos (errores de precio / outliers)

# Ventanas para los análisis rolling
VENTANA = 252   # 1 año hábil de estimación
PASO = 21       # rebalanceo mensual

FIG_DIR = Path(__file__).resolve().parent / "figuras" if "__file__" in globals() else Path("figuras")
FIG_DIR.mkdir(exist_ok=True)

COLOR_PCA = "#eb6834"
COLOR_HPCA = "#2a78d6"
COLOR_MP = "#52514e"
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})


def guardar(fig, nombre):
    fig.savefig(FIG_DIR / f"{nombre}.png", bbox_inches="tight")


# %% 2. Descarga de precios
tickers_locales = [t for lista in SECTORES.values() for t in lista]
tickers_yf = [f"{t}.L" for t in tickers_locales] 

def descargar(tickers, inicio, fin, lote=10, reintentos=3):
    partes, pendientes = [], list(tickers)
    for intento in range(reintentos):
        siguientes = []
        for i in range(0, len(pendientes), lote):
            grupo = pendientes[i:i + lote]
            d = yf.download(grupo, start=inicio, end=fin, auto_adjust=True,
                            progress=False, threads=False)["Close"]
            if isinstance(d, pd.Series):
                d = d.to_frame(grupo[0])
            ok = [t for t in grupo if t in d.columns and d[t].notna().sum() > 0]
            partes.append(d[ok])
            siguientes += [t for t in grupo if t not in ok]
            time.sleep(1.5)
        pendientes = siguientes
        if not pendientes:
            break
        time.sleep(10)          # esperar antes de reintentar los que fallaron
    if pendientes:
        print("No se pudieron bajar:", pendientes)
    return pd.concat(partes, axis=1)


CACHE = Path("precios_uk.csv")
if CACHE.exists():
    precios_raw = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    precios_raw = descargar(tickers_yf, INICIO, FIN)
    precios_raw.to_csv(CACHE)
print(f"Precios descargados: {precios_raw.shape[0]} fechas x {precios_raw.shape[1]} tickers")

# %% 3. Dólar CCL implícito y retornos
# CCL = precio local / (precio ADR / ratio). GGAL: 1 ADR = 10 acciones; YPF: 1 ADR = 1 acción.
# Se usa la mediana de ambas estimaciones para reducir errores puntuales de precio.

precios = precios_raw.reindex(columns=[f"{t}.L" for t in tickers_locales])
precios.columns = tickers_locales

def construir_retornos():
    """Log-retornos diarios en ARS o USD-CCL, sólo en fechas con datos locales y de ADRs."""
    return np.log(precios).diff().iloc[1:]


def limpiar(ret, sectores):
    """Filtra iliquidez/faltantes, winsoriza y ordena las columnas por sector."""
    # La iliquidez se mide en ARS: en USD el CCL hace que un precio sin operar igual "se mueva".
    ret_ars = construir_retornos()
    frac_ceros = (ret_ars == 0).sum() / ret_ars.notna().sum()
    faltantes = ret.isna().mean()
    validas = frac_ceros.index[(frac_ceros <= MAX_FRAC_CEROS) & (faltantes < 0.05)]
    descartadas = sorted(set(ret.columns) - set(validas))
    if descartadas:
        print(f"Descartadas por iliquidez/faltantes: {descartadas}")

    sectores_ok = {s: [t for t in ts if t in validas] for s, ts in sectores.items()}
    sectores_ok = {s: ts for s, ts in sectores_ok.items() if len(ts) >= 2}
    for s, ts in sectores_ok.items():
        if len(ts) < 3:
            print(f"Atención: el sector '{s}' quedó con sólo {len(ts)} acciones")

    orden = [t for ts in sectores_ok.values() for t in ts]
    r = ret[orden].dropna()
    mu, sd = r.mean(), r.std()
    r = r.clip(mu - WINSOR_SIGMA * sd, mu + WINSOR_SIGMA * sd, axis=1)
    return r, sectores_ok


retornos_raw = construir_retornos()
retornos, sectores = limpiar(retornos_raw, SECTORES)
etiquetas = np.array([s for s, ts in sectores.items() for _ in ts])
nombres_sector = list(sectores)
T, n = retornos.shape
print(f"\nMuestra final ({MONEDA}): T = {T} días, n = {n} acciones, b = {len(sectores)} sectores")
print(f"Período: {retornos.index[0].date()} a {retornos.index[-1].date()}")
for s, ts in sectores.items():
    print(f"  {s:<20} ({len(ts)}): {', '.join(ts)}")


# %% 4. Funciones: PCA y HPCA
def estandarizar(r):
    return ((r - r.mean()) / r.std()).to_numpy()


def eig_desc(A):
    """Autovalores/autovectores de una matriz simétrica, en orden decreciente."""
    val, vec = np.linalg.eigh(A)
    return val[::-1], vec[:, ::-1]


def orientar(v):
    """Fija el signo de un autovector para que la suma de sus entradas sea positiva."""
    return v if v.sum() >= 0 else -v


def hpca(X, etiquetas):
    """
    HPCA de dos niveles (Avellaneda 2019).

    X: retornos estandarizados (T x n), columnas ordenadas por sector.
    Devuelve un dict con la matriz HPCA R~, los factores sectoriales, rho_bar, M y la
    descomposición espectral analítica (Proposición 2) con etiqueta por autovector.
    """
    T, n = X.shape
    R = X.T @ X / (T - 1)
    sectores = list(dict.fromkeys(etiquetas))
    idx = {s: np.where(etiquetas == s)[0] for s in sectores}

    beta = np.zeros(n)
    lam1 = np.zeros(len(sectores))
    F = np.zeros((T, len(sectores)))
    W1 = np.zeros((n, len(sectores)))   # W^(1,k): EV1 de cada sector embebido en R^n (ec. 13)
    intra = []                           # (lambda^(j,k), W^(j,k), sector) para j >= 2

    for k, s in enumerate(sectores):
        ii = idx[s]
        val, vec = eig_desc(R[np.ix_(ii, ii)])
        v1 = orientar(vec[:, 0])
        lam1[k] = val[0]
        beta[ii] = np.sqrt(val[0]) * v1                  # ec. 7
        F[:, k] = X[:, ii] @ v1 / np.sqrt(val[0])        # ec. 5
        W1[ii, k] = v1
        for j in range(1, len(ii)):
            w = np.zeros(n)
            w[ii] = vec[:, j]
            intra.append((val[j], w, s))

    rho_bar = np.corrcoef(F, rowvar=False)
    sec_id = np.array([sectores.index(s) for s in etiquetas])
    mismo = sec_id[:, None] == sec_id[None, :]
    R_tilde = np.where(mismo, R, np.outer(beta, beta) * rho_bar[np.ix_(sec_id, sec_id)])  # ec. 11

    # Proposición 2: espectro analítico
    M = np.sqrt(np.outer(lam1, lam1)) * rho_bar           # ec. 15
    mu, alpha = eig_desc(M)
    alpha[:, 0] = orientar(alpha[:, 0])
    espectro = [(mu[k], W1 @ alpha[:, k], "Multi-sector", alpha[:, k]) for k in range(len(sectores))]  # ec. 16
    espectro += [(lam, w, s, None) for lam, w, s in intra]
    espectro.sort(key=lambda e: -e[0])

    return {
        "R": R, "R_tilde": R_tilde, "beta": beta, "F": F, "lam1": lam1,
        "rho_bar": rho_bar, "M": M, "sectores": sectores,
        "valores": np.array([e[0] for e in espectro]),
        "vectores": np.column_stack([e[1] for e in espectro]),
        "tipo": [e[2] for e in espectro],
        "alpha": [e[3] for e in espectro],
    }


def verificar_hpca(h, tol=1e-8):
    """Chequeos numéricos de las Proposiciones 1 y 2."""
    Rt = h["R_tilde"]
    val_num = eig_desc(Rt)[0]
    checks = {
        "R~ simétrica": np.allclose(Rt, Rt.T, atol=tol),
        "diag(R~) = 1": np.allclose(np.diag(Rt), 1, atol=1e-6),
        "R~ semidefinida positiva (Prop. 1)": val_num.min() > -tol,
        "espectro numérico = analítico (Prop. 2)": np.allclose(val_num, h["valores"], atol=1e-6),
        "autovectores analíticos verifican R~ v = λ v": np.allclose(
            Rt @ h["vectores"], h["vectores"] * h["valores"], atol=1e-6),
    }
    for nombre, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {nombre}")
    return all(checks.values())


# %% 5. Estimación en la muestra completa
X = estandarizar(retornos)
h = hpca(X, etiquetas)
R, R_tilde = h["R"], h["R_tilde"]

lam_pca, V_pca = eig_desc(R)
lam_hpca, V_hpca = h["valores"], h["vectores"]
V_pca[:, 0] = orientar(V_pca[:, 0])
V_hpca[:, 0] = orientar(V_hpca[:, 0])

print("Verificación de la implementación:")
verificar_hpca(h)

print("\nCorrelación entre factores sectoriales (rho_bar):")
print(pd.DataFrame(h["rho_bar"], index=h["sectores"], columns=h["sectores"]).round(2).to_string())

lam_mp = (1 + np.sqrt(n / T)) ** 2
m_rmt = int((lam_pca > lam_mp).sum())
print(f"\nCota Marchenko-Pastur λ+ = (1+√(n/T))² = {lam_mp:.2f}")
print(f"Autovalores de R por encima de λ+: {m_rmt} → factores 'significativos' según RMT")
print(f"Número de condición: R = {lam_pca[0] / lam_pca[-1]:.1f} | R~ = {lam_hpca[0] / lam_hpca[-1]:.1f}")

# %% 6. Varianza explicada (Fig. 1 del paper)
k = np.arange(1, n + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(k, np.cumsum(lam_pca) / n, color=COLOR_PCA, lw=2, marker="o", ms=4, label="PCA")
ax.plot(k, np.cumsum(lam_hpca) / n, color=COLOR_HPCA, lw=2, marker="o", ms=4, label="HPCA")
ax.set(xlabel="Cantidad de factores k", ylabel="Varianza explicada acumulada",
       title=f"Varianza explicada: PCA vs HPCA ({MONEDA})", ylim=(0, 1.02))
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.legend(frameon=False)
guardar(fig, "01_varianza_explicada")
plt.show()

print(f"EV1 explica {lam_pca[0] / n:.1%} (PCA) vs {lam_hpca[0] / n:.1%} (HPCA); "
      f"diferencia = {(lam_pca[0] - lam_hpca[0]) / n:.2%}")


# %% 7. Tabla de autovalores e interpretación (Tabla 2 del paper)
def describir(tipo, alpha, nombres, umbral=0.3):
    """Interpretación de un eigenportfolio HPCA."""
    if tipo != "Multi-sector":
        return f"Long-short dentro de {tipo}"
    if np.all(alpha > 0):
        return "Mercado (long en todos los sectores)"
    largos = [nombres[i] for i in np.where(alpha > umbral)[0]]
    cortos = [nombres[i] for i in np.where(alpha < -umbral)[0]]
    return f"Multi-sector: +[{', '.join(largos)}] vs −[{', '.join(cortos)}]"


tabla = pd.DataFrame({
    "λ PCA": lam_pca,
    "λ HPCA": lam_hpca,
    "% var PCA": lam_pca / n,
    "% var HPCA": lam_hpca / n,
    "Eigenportfolio HPCA": [describir(t, a, h["sectores"]) for t, a in zip(h["tipo"], h["alpha"])],
}, index=pd.RangeIndex(1, n + 1, name="k"))
with pd.option_context("display.max_colwidth", 80, "display.width", 200):
    print(tabla.head(15).to_string(formatters={
        "λ PCA": "{:.2f}".format, "λ HPCA": "{:.2f}".format,
        "% var PCA": "{:.1%}".format, "% var HPCA": "{:.1%}".format}))

N_EV = min(25, n)     # autovectores a analizar
N_LINEAS = 5          # sólo estos van como gráfico de líneas

for j in range(N_EV):
    if V_hpca[:, j] @ V_pca[:, j] < 0:  # alinea signos para comparar
        V_pca[:, j] *= -1
similitud = np.abs(np.sum(V_pca[:, :N_EV] * V_hpca[:, :N_EV], axis=0))
print("Similitud coseno |<v_PCA, v_HPCA>| por autovector:")
for j, s in enumerate(similitud, 1):
    print(f"  EV{j}: {s:.3f}   ({tabla['Eigenportfolio HPCA'].iloc[j - 1]})")
print(f"Correlación de las series de retornos del EV1 PCA vs HPCA: "
      f"{np.corrcoef(X @ V_pca[:, 0], X @ V_hpca[:, 0])[0, 1]:.4f}")


def sombrear_sectores(ax, con_texto=True):
    inicio = 0
    for i, s in enumerate(h["sectores"]):
        fin = inicio + int((etiquetas == s).sum())
        if i % 2 == 0:
            ax.axvspan(inicio - 0.5, fin - 0.5, color="#000000", alpha=0.05, lw=0)
        if con_texto:
            ax.text((inicio + fin - 1) / 2, 1.0, s, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color="#52514e")
        inicio = fin


# --- 1) Líneas: sólo los primeros N_LINEAS autovectores ---
x = np.arange(n)
fig, axes = plt.subplots(N_LINEAS, 1, figsize=(20, 2.6 * N_LINEAS), sharex=True)
for j, ax in enumerate(axes):
    sombrear_sectores(ax, con_texto=(j == 0))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.plot(x, V_pca[:, j], color=COLOR_PCA, lw=1.5, marker="o", ms=3, label="PCA")
    ax.plot(x, V_hpca[:, j], color=COLOR_HPCA, lw=1.5, marker="o", ms=3, label="HPCA")
    ax.set_ylabel(f"EV{j + 1}")
    ax.set_title(f"EV{j + 1} — similitud coseno {similitud[j]:.2f}", loc="left",
                 fontsize=10, pad=14 if j == 0 else 4)
axes[0].legend(frameon=False, loc="upper right", ncols=2)
axes[-1].set_xticks(x, retornos.columns, rotation=90, fontsize=5)
fig.suptitle(f"Primeros {N_LINEAS} autovectores: PCA vs HPCA ({MONEDA})", y=1.0)
fig.tight_layout()
guardar(fig, "02_autovectores")
plt.show()

# --- 2) Solapamiento: |<v_PCA_i, v_HPCA_j>| ---
O = np.abs(V_pca[:, :N_EV].T @ V_hpca[:, :N_EV])     # filas = PCA, columnas = HPCA
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(O, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(np.arange(N_EV), np.arange(1, N_EV + 1), fontsize=7)
ax.set_yticks(np.arange(N_EV), np.arange(1, N_EV + 1), fontsize=7)
ax.set(xlabel="Autovector HPCA", ylabel="Autovector PCA",
       title=f"Solapamiento PCA vs HPCA ({MONEDA})")
ax.grid(False)
fig.colorbar(im, ax=ax, label="|coseno|")
fig.tight_layout()
guardar(fig, "02b_solapamiento")
plt.show()

# Mejor correspondencia de cada autovector HPCA entre TODOS los autovectores PCA
O_full = np.abs(V_pca.T @ V_hpca[:, :N_EV])
print("Mejor match PCA para cada autovector HPCA (índice PCA, |coseno|):")
for j in range(N_EV):
    i = O_full[:, j].argmax()
    print(f"  HPCA EV{j + 1} ({tabla['Eigenportfolio HPCA'].iloc[j]}) -> PCA EV{i + 1}, {O_full[i, j]:.2f}")

# --- 3) Concentración por sector ---
sec_list = h["sectores"]
colores = plt.cm.tab20(np.arange(len(sec_list)) % 20)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
for ax, V, nombre in [(axes[0], V_pca, "PCA"), (axes[1], V_hpca, "HPCA")]:
    M = np.array([[np.sum(V[etiquetas == s, j] ** 2) for j in range(N_EV)] for s in sec_list])
    base = np.zeros(N_EV)
    for i, s in enumerate(sec_list):
        ax.bar(np.arange(1, N_EV + 1), M[i], bottom=base, color=colores[i], label=s, width=0.8)
        base += M[i]
    ax.set(title=nombre, xlabel="Autovector")
axes[0].set_ylabel("Fracción de ‖v‖² por sector")
axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
fig.suptitle(f"¿Cuánto pesa cada sector en cada autovector? ({MONEDA})")
fig.tight_layout()
guardar(fig, "02c_concentracion_sector")
plt.show()

# %% 9. Matrices de correlación: empírica, HPCA y diferencia
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (titulo, A, cmap, lim) in zip(axes, [
    ("R empírica", R, "Blues", (0, 1)),
    ("R~ HPCA", R_tilde, "Blues", (0, 1)),
    ("R − R~ (correlación que HPCA asume nula)", R - R_tilde, "RdBu_r", (-0.3, 0.3)),
]):
    im = ax.imshow(A, cmap=cmap, vmin=lim[0], vmax=lim[1])
    ax.set_title(titulo, fontsize=10)
    tam = [int((etiquetas == s).sum()) for s in h["sectores"]]
    centros = np.cumsum(tam) - np.array(tam) / 2 - 0.5
    ax.set_xticks(centros, h["sectores"], rotation=90, fontsize=8)
    ax.set_yticks(centros, h["sectores"], fontsize=8)
    ax.grid(False)
    limites = np.cumsum([int((etiquetas == s).sum()) for s in h["sectores"]])[:-1] - 0.5
    for b in limites:
        ax.axhline(b, color="#0b0b0b", lw=0.6)
        ax.axvline(b, color="#0b0b0b", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
guardar(fig, "03_matrices_correlacion")
plt.show()

fuera = etiquetas[:, None] != etiquetas[None, :]
print(f"Correlación entre sectores: media empírica {R[fuera].mean():.3f} vs HPCA {R_tilde[fuera].mean():.3f}; "
      f"error absoluto medio {np.abs(R - R_tilde)[fuera].mean():.3f}")


# %% 10. Residuos vs Marchenko-Pastur (Sección 6 / Fig. 14 del paper)
def residuos(X, V):
    """Residuos de regresar X sobre los factores F = X V (MCO)."""
    F = X @ V
    coef, *_ = np.linalg.lstsq(F, X, rcond=None)
    return X - F @ coef


def densidad_mp(q, lam):
    lmin, lmax = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    d = np.zeros_like(lam)
    dentro = (lam > lmin) & (lam < lmax)
    d[dentro] = np.sqrt((lmax - lam[dentro]) * (lam[dentro] - lmin)) / (2 * np.pi * q * lam[dentro])
    return d


# Se remueven m = b factores: para HPCA es justo el subespacio Ω de factores multi-sector (ec. 14).
# (El paper usa m = 30 sobre n = 434; con n ≈ 27 usar m_rmt = 1 dejaría la comparación trivial.)
m = len(sectores)
eps_pca = residuos(X, V_pca[:, :m])
eps_hpca = residuos(X, V_hpca[:, :m])
# Los residuos tienen rango n − m: se descartan los m autovalores nulos
lam_res_pca = eig_desc(np.corrcoef(eps_pca, rowvar=False))[0][:n - m]
lam_res_hpca = eig_desc(np.corrcoef(eps_hpca, rowvar=False))[0][:n - m]
print(f"Removiendo m = {m} factores:")
print(f"  Máx. autovalor de residuos: PCA {lam_res_pca[0]:.2f} | HPCA {lam_res_hpca[0]:.2f} | λ+ MP {lam_mp:.2f}")
print(f"  Autovalores sobre λ+: PCA {(lam_res_pca > lam_mp).sum()} | HPCA {(lam_res_hpca > lam_mp).sum()}")

# Con n ≈ 27 un histograma es poco informativo: se grafican los autovalores ordenados (scree)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
kk = np.arange(1, n - m + 1)
ax.plot(kk, lam_res_pca, color=COLOR_PCA, lw=2, marker="o", ms=5, label="Residuos PCA")
ax.plot(kk, lam_res_hpca, color=COLOR_HPCA, lw=2, marker="o", ms=5, label="Residuos HPCA")
ax.axhline(lam_mp, color=COLOR_MP, ls="--", lw=1)
ax.text(n - m, lam_mp, f"λ+ MP = {lam_mp:.2f} ", color=COLOR_MP, fontsize=9, ha="right", va="bottom")
ax.set(xlabel="Rango", ylabel="Autovalor", title=f"Autovalores de residuos (m = {m} factores removidos)")
ax.legend(frameon=False)

ax = axes[1]
bins = np.linspace(0, max(lam_res_pca[0], lam_res_hpca[0], lam_mp) * 1.05, 20)
ax.hist([lam_res_pca, lam_res_hpca], bins=bins, density=True, color=[COLOR_PCA, COLOR_HPCA],
        label=["Residuos PCA", "Residuos HPCA"], edgecolor="white", linewidth=1)
grid = np.linspace(1e-3, bins[-1], 400)
ax.plot(grid, densidad_mp(n / T, grid), color=COLOR_MP, lw=2, label="Densidad Marchenko-Pastur")
ax.set(xlabel="Autovalor", ylabel="Densidad", title="Distribución vs Marchenko-Pastur")
ax.legend(frameon=False)
fig.suptitle(f"Análisis de residuos vía RMT ({MONEDA})")
fig.tight_layout()
guardar(fig, "04_residuos_mp")
plt.show()


# %% 11. Estabilidad temporal de los autovectores (problema de identificación)
def solapamiento_subespacio(A, B):
    """Promedio de cos² de los ángulos principales entre span(A) y span(B); 1 = mismo subespacio."""
    return np.linalg.norm(A.T @ B, "fro") ** 2 / A.shape[1]


K_EST = 5
filas = []
previo = None
for fin in range(VENTANA, T + 1, PASO):
    Xw = estandarizar(retornos.iloc[fin - VENTANA:fin])
    _, Vp = eig_desc(np.corrcoef(Xw, rowvar=False))
    hw = hpca(Xw, etiquetas)
    actual = (Vp[:, :K_EST], hw["vectores"][:, :K_EST])
    if previo is not None:
        fila = {"fecha": retornos.index[fin - 1]}
        for nombre, A, B in [("PCA", previo[0], actual[0]), ("HPCA", previo[1], actual[1])]:
            # "EVj": mismo rango; "EVj*": mejor match entre los top-K (tolera cambios de orden
            # entre autovalores casi iguales, que en HPCA son frecuentes)
            O = np.abs(A.T @ B)
            for j in range(K_EST):
                fila[f"{nombre} EV{j + 1}"] = O[j, j]
                fila[f"{nombre} EV{j + 1}*"] = O[j].max()
            fila[f"{nombre} sub3"] = solapamiento_subespacio(A[:, :3], B[:, :3])
            fila[f"{nombre} sub5"] = solapamiento_subespacio(A, B)
        filas.append(fila)
    previo = actual

estab = pd.DataFrame(filas).set_index("fecha")
cols = [f"EV{j}" for j in range(1, K_EST + 1)] + [f"EV{j}*" for j in range(1, K_EST + 1)] + ["sub3", "sub5"]
resumen = pd.DataFrame({met: [estab[f"{met} {c}"].mean() for c in cols] for met in ["PCA", "HPCA"]},
                       index=cols[:-2] + ["Subespacio top-3", "Subespacio top-5"])
print(f"Estabilidad mes a mes (ventana {VENTANA}d, paso {PASO}d) — 1 = idéntico; * = mejor match:")
print(resumen.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, col, titulo in [(axes[0], "EV2", "Autovector 2"), (axes[1], "sub5", "Subespacio de los 5 primeros")]:
    ax.plot(estab.index, estab[f"PCA {col}"], color=COLOR_PCA, lw=2, label="PCA")
    ax.plot(estab.index, estab[f"HPCA {col}"], color=COLOR_HPCA, lw=2, label="HPCA")
    ax.set(title=f"Estabilidad: {titulo}", ylim=(0, 1.02))
axes[0].set_ylabel("Solapamiento con el mes anterior")
axes[0].legend(frameon=False)
fig.tight_layout()
guardar(fig, "05_estabilidad")
plt.show()


# %% 12. Cartera de mínima varianza out-of-sample
def corr_pca_truncada(R, m):
    """m factores PCA + varianza idiosincrática diagonal (modelo factorial de la ec. 8)."""
    val, vec = eig_desc(R)
    C = (vec[:, :m] * val[:m]) @ vec[:, :m].T
    np.fill_diagonal(C, 1.0)
    return C


def pesos_min_var(C):
    unos = np.ones(len(C))
    w = np.linalg.solve(C, unos)
    return w / w.sum()


estimadores = {
    "Muestral": lambda Xw: np.corrcoef(Xw, rowvar=False),
    f"PCA ({m} factores)": lambda Xw: corr_pca_truncada(np.corrcoef(Xw, rowvar=False), m),
    "HPCA": lambda Xw: hpca(Xw, etiquetas)["R_tilde"],
    "Ledoit-Wolf": lambda Xw: (lambda S: S / np.sqrt(np.outer(np.diag(S), np.diag(S))))(
        LedoitWolf().fit(Xw).covariance_),
}

rend = {k: [] for k in list(estimadores) + ["Equiponderada"]}
pesos_previos = {k: None for k in rend}
rotacion = {k: [] for k in rend}
apalancamiento = {k: [] for k in rend}
r_np = retornos.to_numpy()
for ini in range(VENTANA, T - PASO + 1, PASO):
    ventana = retornos.iloc[ini - VENTANA:ini]
    sd = ventana.std().to_numpy()
    Xw = estandarizar(ventana)
    pesos = {}
    for nombre, est in estimadores.items():
        C = est(Xw)
        Sigma = C * np.outer(sd, sd)
        pesos[nombre] = pesos_min_var(Sigma)
    pesos["Equiponderada"] = np.full(n, 1 / n)
    for nombre, w in pesos.items():
        rend[nombre].append(r_np[ini:ini + PASO] @ w)
        apalancamiento[nombre].append(np.abs(w).sum())
        if pesos_previos[nombre] is not None:
            rotacion[nombre].append(np.abs(w - pesos_previos[nombre]).sum())
        pesos_previos[nombre] = w

fechas_oos = retornos.index[VENTANA:VENTANA + len(np.concatenate(rend["Muestral"]))]
rend = pd.DataFrame({k: np.concatenate(v) for k, v in rend.items()}, index=fechas_oos)
res_mv = pd.DataFrame({
    "Vol. anual realizada": rend.std() * np.sqrt(252),
    "Retorno anual": rend.mean() * 252,
    "Exposición bruta Σ|w|": pd.Series({k: np.mean(v) for k, v in apalancamiento.items()}),
    "Rotación mensual": pd.Series({k: np.mean(v) for k, v in rotacion.items()}),
})
res_mv["Ret/Vol"] = res_mv["Retorno anual"] / res_mv["Vol. anual realizada"]
print(f"Mínima varianza out-of-sample ({fechas_oos[0].date()} a {fechas_oos[-1].date()}, {MONEDA}):")
print(res_mv.sort_values("Vol. anual realizada").round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 4.5))
colores_mv = {"Muestral": "#4a3aa7", f"PCA ({m} factores)": COLOR_PCA, "HPCA": COLOR_HPCA,
              "Ledoit-Wolf": "#1baf7a", "Equiponderada": "#52514e"}
for nombre, serie in rend.items():
    ax.plot(serie.index, np.exp(serie.cumsum()), lw=2, color=colores_mv[nombre], label=nombre)
ax.set(title=f"Carteras de mínima varianza out-of-sample (base 1, {MONEDA})", ylabel="Valor")
ax.legend(frameon=False, ncols=2)
guardar(fig, "06_min_varianza_oos")
plt.show()


# %% 13. Robustez: GBP vs USD — ¿el primer factor es "mercado" o "libra"?
def bajar_serie(ticker):
    s = yf.download(ticker, start=INICIO, end=FIN, auto_adjust=True,
                    progress=False, threads=False)["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


ftse = bajar_serie("^FTSE")
gbpusd = bajar_serie("GBPUSD=X")

# El FX cotiza en un calendario distinto: se alinea a las fechas de las acciones
def dlog(serie):
    return np.log(serie).reindex(precios.index).ffill().diff().reindex(retornos_raw.index)


r_fx = dlog(gbpusd)
r_ftse_gbp = dlog(ftse)
retornos_usd = retornos_raw.add(r_fx, axis=0)   # log(P·FX) = log P + log FX
r_ftse_usd = r_ftse_gbp + r_fx

casos = {"GBP": (retornos_raw, r_ftse_gbp), "USD": (retornos_usd, r_ftse_usd)}
comparacion = {}
for moneda, (ret_m, r_idx) in casos.items():
    r_m, sec_m = limpiar(ret_m, SECTORES)
    et_m = np.array([s for s, ts in sec_m.items() for _ in ts])
    X_m = estandarizar(r_m)
    h_m = hpca(X_m, et_m)
    f1 = pd.Series(X_m @ orientar(h_m["vectores"][:, 0]), index=r_m.index)
    lam_m = eig_desc(h_m["R"])[0]
    comparacion[moneda] = {
        "EV1 % var PCA": lam_m[0] / X_m.shape[1],
        "EV1 % var HPCA": h_m["valores"][0] / X_m.shape[1],
        "Corr(EV1 HPCA, FTSE 100)": f1.corr(r_idx),
        "Corr(EV1 HPCA, Δ log GBPUSD)": f1.corr(r_fx),
        "Corr. media entre sectores": h_m["rho_bar"][np.triu_indices(len(sec_m), 1)].mean(),
    }
print(pd.DataFrame(comparacion).round(3).to_string())

# %% 14. Robustez: 3 vs 11 sectores
for nombre, part in [("3 sectores", SECTORES_3), ("11 sectores", SECTORES)]:
    r_p, sec_p = limpiar(retornos_raw, part)
    et_p = np.array([s for s, ts in sec_p.items() for _ in ts])
    X_p = estandarizar(r_p)
    h_p = hpca(X_p, et_p)
    lam_p = eig_desc(h_p["R"])[0]
    fuera_p = et_p[:, None] != et_p[None, :]
    print(f"{nombre}: EV1 PCA {lam_p[0] / len(lam_p):.1%} | EV1 HPCA {h_p['valores'][0] / len(lam_p):.1%} | "
          f"error medio fuera de bloque {np.abs(h_p['R'] - h_p['R_tilde'])[fuera_p].mean():.3f} | "
          f"factores multi-sector en top-5: {h_p['tipo'][:5].count('Multi-sector')}")
