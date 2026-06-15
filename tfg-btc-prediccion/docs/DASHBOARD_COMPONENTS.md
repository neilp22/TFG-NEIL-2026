# Dashboard — Inventario de componentes (para Claude Code design pass)

**Archivo**: `agente_ia/templates/dashboard.html` (~2000 líneas, single-file HTML+CSS+JS)
**Stack**: Vanilla HTML5/CSS/JS, sin framework. TradingView Lightweight Charts v4.2.0. Flask backend.
**Constraints**: NO romper IDs ni `data-*` attributes (JS depende de ellos). NO añadir frameworks externos. NO tocar lógica JS.

---

## 🎨 Design system actual (CSS variables ya definidas en `:root`)

```css
/* Backgrounds (4 capas) */
--bg-base:        #0A0D13;
--bg-surface:     #0F1319;
--bg-elevated:    #161B24;
--bg-overlay:     #1C2333;

/* Accent + semantic */
--accent:         #00C4FF;       --accent-dim:     rgba(0,196,255,0.15);
--accent-glow:    0 0 12px rgba(0,196,255,0.33);
--color-up:       #22C55E;
--color-down:     #EF4444;
--color-neutral:  #94A3B8;
--color-warning:  #F59E0B;

/* Text */
--text-primary:   #E2E8F0;
--text-secondary: #94A3B8;
--text-muted:     #475569;
--text-accent:    #00C4FF;

/* Borders */
--border-subtle:  #1E2D40;
--border-default: #243447;
--border-accent:  rgba(0,196,255,0.25);

/* Glassmorphism card */
--card-bg:        rgba(15, 19, 25, 0.85);
--card-border:    1px solid #1E2D40;
--card-blur:      blur(12px);
--card-shadow:    0 4px 24px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.03) inset;

/* Typography */
--font-mono:      'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
--font-sans:      'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
--radius:         8px;
--radius-pill:    20px;

/* Aliases legacy (JS usa) */
--green = --color-up, --red = --color-down, --blue = --accent
--yellow = --color-warning, --bg-primary = --bg-base, etc.
```

**Aesthetic target**: Bloomberg Terminal + Fintech moderno (glassmorphism + neón cyan).

---

## 📐 Layout overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ #topbar (48px) — height fija                                             │
│ [BTC/USDT pill] [PRICE] [24h Δ chip] | [High] [Low] [Vol] | [Spread]    │
│ [Sesion]                                              [● LIVE pulse]    │
├──────────────────────────────────────┬──────────────────────────────────┤
│ #chart-panel (flexible width)        │ #right-panel (680px fijo)        │
│ ┌──────────────────────────────────┐ │ ┌──────────────────────────────┐ │
│ │ #chart-controls                  │ │ │ #quick-stats (3 cards)       │ │
│ │ TF: [15m][1h][4h][1D]            │ │ │ Confluencia | Capital | Bot  │ │
│ │ Overlays: OB FVG Swings KZ       │ │ ├──────────────────────────────┤ │
│ ├──────────────────────────────────┤ │ │ #conf-bars-mini              │ │
│ │ #chart-container (TradingView)   │ │ │ 6 mini bars módulos          │ │
│ │ Velas + indicadores + overlays   │ │ ├──────────────────────────────┤ │
│ │ Volumen (parte inferior)         │ │ │ #pos-card (open position)    │ │
│ │ Eje X en hora local (es-ES)      │ │ ├──────────────────────────────┤ │
│ │                                  │ │ │ #right-tab-nav (8 tabs)      │ │
│ │                                  │ │ ├──────────────────────────────┤ │
│ │                                  │ │ │ #right-tab-content (scroll)  │ │
│ │                                  │ │ │ Contenido del tab activo     │ │
│ └──────────────────────────────────┘ │ └──────────────────────────────┘ │
└──────────────────────────────────────┴──────────────────────────────────┘
```

Responsive: `1fr 680px` default · `1fr 620px` <1500px · `1fr 560px` <1280px.

---

## 📦 Componentes detallados

### A. TOPBAR (`#topbar`, 48px)

| ID/clase | Contenido | Data source | Refresh |
|---|---|---|---|
| `.sym` | "BTC / USDT" pill cyan | estático | — |
| `#price-display` | Precio grande mono 22px | `/api/live.price` | 5s |
| `#tb-change` | Δ 24h con color bull/bear | `/api/live.change_24h` | 5s |
| `#tb-high` | High 24h chip | `/api/live.high_24h` | 5s |
| `#tb-low` | Low 24h chip | `/api/live.low_24h` | 5s |
| `#tb-vol` | Vol 24h chip (BTC units) | `/api/live.volume_24h` | 5s |
| `#tb-spread` | Spread chip | `/api/live.spread` | 5s |
| `#tb-session` | Asia/London/NY + [KZ] badge | `/api/session.current_session` | 30s |
| `#ws-dot` | Punto pulsante cyan = conectado | `/api/live.source` | 5s |
| `#ws-label` | "LIVE" | estático | — |

**Mejoras visuales sugeridas**:
- Sticky positioning ya OK
- Pills topbar actuales son OK, posible mejora: añadir mini sparkline al precio
- Indicador de "última actualización hace Xs"

---

### B. CHART PANEL (`#chart-panel`)

**Controles** (`#chart-controls`, padding 8px 12px):
- `.tf-btn[data-tf="15m|1h|4h|1d"]` — botones timeframe (active = fondo cyan)
- `.chart-sep` — divider vertical
- `.layer-btn` — toggles overlays: `#btn-ob`, `#btn-fvg`, `#btn-swing`, `#btn-kz`
  - Active = fondo `--accent-dim` + texto `--accent`

**Chart** (`#chart-container`, TradingView Lightweight Charts):
- Velas (`addCandlestickSeries`) — upColor verde, downColor rojo
- Volumen como histograma — gris translúcido, `priceScaleId=''`, scaleMargins separados
- Overlays dinámicos:
  - Order Blocks (líneas dashed verde/rojo, ±100 price zones)
  - FVG (líneas dotted)
  - Swing highs/lows (líneas finas semi-transparentes)
  - Killzone markers (círculos amber `aboveBar`)
  - News markers (flechas verde/rojo según sentiment)
  - Trade activo (líneas Entry/SL/TP)
- Zoom inicial: últimas 96 velas (15m) / 120 (1h) / 90 (4h) / 120 (1d)
- Eje X formateado en hora local ES (`tickMarkFormatter`)

**Mejoras visuales sugeridas**:
- Crosshair tooltip más rico (OHLCV + indicators del momento)
- Mini info bar arriba del chart con last close + change
- Botón "fit content" / "reset zoom"
- Volumen con buckets de color verde/rojo según direction

---

### C. RIGHT PANEL — Quick Stats (`#quick-stats`, grid 3 cards 10px gap)

| ID | Contenido | Data source |
|---|---|---|
| `#qs-conf-score` | Score numérico ±X.XXX colored | `/api/signals/breakdown.final.score` |
| `#qs-conf-label` | "BUY/SELL/NEUTRAL" colored | `/api/signals/breakdown.final.label` |
| `#qs-capital` | Capital actual $XX,XXX mono | `/api/bot/metrics.capital_current` |
| `#qs-return` | Retorno % bull/bear | `/api/bot/metrics.total_return_pct` |
| `#qs-bot-status` | "activo/detenido" colored | `/api/bot/status.is_running` |
| `#qs-trades` | "N trades" muted | `/api/bot/metrics.closed_trades` |

**Estilo**: `.qs-block` con `var(--card-bg)`, hover → `--border-accent` + glow.

**Mejoras sugeridas**: sparklines pequeños en cada card (mini trend last 7 days).

---

### D. CONFLUENCE BARS (`#conf-bars-mini`)

Render dinámico de 6 filas (una por módulo):
```html
<div class="conf-row">
  <span class="conf-label-mod">TECNICO</span>
  <div class="conf-bar-wrap">  <!-- track gris -->
    <div class="conf-center-line"></div>   <!-- vertical center -->
    <div class="conf-bar pos|neg" style="width:X%"></div>   <!-- gradient -->
  </div>
  <span class="conf-val">+0.42</span>
</div>
```
Módulos: TECNICO, ICT, MTF, SMARTMONEY, SENTIMIENTO, ML MODEL (orden fijo)
Data: `/api/signals/breakdown.final.scores_by_module` (dict)
Refresh: 60s

**Mejoras sugeridas**:
- Animación al cambiar valores
- Tooltip con detalles de cada módulo (signals que se activaron)
- Icono por módulo

---

### E. POSITION CARD (`#pos-card`)

```html
<div id="pos-card">
  <span id="pos-card-label">Posicion abierta</span>
  <div id="pos-content">...</div>
</div>
```

Estados:
- **Sin posición**: `class="pos-no-trade"`, texto centrado muted, border-left subtle
- **Long open**: border-left verde 3px, dir-badge verde, filas Entry/SL/TP/PnL
- **Short open**: border-left rojo 3px, dir-badge rojo

CSS usa `:has(.dir-long)` / `:has(.dir-short)` para colorear el borde.

Data: `/api/bot/status.open_positions[0]`
Refresh: 15s

**Mejoras sugeridas**:
- Barra de progreso visual SL → Entry → TP
- Tiempo restante hasta timeout
- Indicador en tiempo real del PnL animado

---

### F. TAB NAVIGATION (`#right-tab-nav`, 8 botones)

| `data-tab` | Label | Icono SVG (Lucide) | Función JS |
|---|---|---|---|
| `resumen` | Resumen | BarChart2 | `refreshResumen()` |
| `senales` | Señales | Activity | `refreshSenales()` |
| `sentimiento` | Sentiment | MessageSquare | `refreshSentimiento()` |
| `agente` | Agente IA | Bot | `() => {}` (lazy) |
| `bot` | Bot | Settings2 | `refreshBot()` |
| `sistema` | Sistema | Shield | `refreshSistema()` |
| `noticias` | Noticias | Newspaper | `loadNoticias()` |
| `educacion` | Educación | GraduationCap | `() => {}` (estático) |

Active: border-bottom 2px cyan + color text-accent + background bg-elevated.

**Mejoras sugeridas**:
- Badge en tabs con count (e.g., Noticias: "12 nuevas")
- Mejor truncamiento si overflow horizontal

---

### G. TAB CONTENT — detalle de cada pane

#### G.1 `#rtab-resumen` (tab Resumen)

Componentes:
- `#resumen-cards` — grid 3 columnas con 6 cards (RSI 1h, RSI 4h, Vol Ratio, ATR, Estructura, Killzone)
- Tabla MTF (`#resumen-tbody`) con cols: TF / Estr / RSI / EMA / MACD / BB / VWAP / Vol
- Alertas ICT (`#alerts-list`) — items con border-left según dirección, distancia %

Data: `/api/market` + `/api/ict` + `/api/bot/alerts` (dedup top 12)

#### G.2 `#rtab-senales` (tab Señales)

- `#signals-modules` — 6 barras horizontales (una por módulo) con valor y porcentaje
- Tabla niveles técnicos (`#levels-tbody`) — Precio / Tipo / TF / Dist% / R:R

Data: `/api/signals/breakdown` + `/api/levels`

#### G.3 `#rtab-sentimiento` (tab Sentiment)

Grid 2 columnas:
- Izquierda: FinBERT card grande + barras 7d
- Derecha: Fear & Greed canvas gauge + history 5 días
- Abajo: tabla últimas noticias con score colored

Data: `/api/sentiment` + `/api/fear_greed` + `/api/system/transparency.top_articles`

#### G.4 `#rtab-agente` (tab Agente IA) — flex column 100% height

- Botón `#agent-btn` "▶ Analizar Mercado Ahora" (azul prominente)
- Spinner `#agent-spinner` "⟳ Analizando..."
- `#agent-output` — div scrollable que renderiza HTML del agente streamed via SSE

Output del agente incluye: h3 sections, tables, setup-box (valid/invalid), span class bull/bear.

**Mejoras sugeridas para esta tab**:
- Historial de análisis previos (timestamp + score)
- Botón "Regenerar"
- Indicador de costo de la llamada
- Modo "expandir a fullscreen"

#### G.5 `#rtab-bot` (tab Bot)

- 6 cards métricas (Capital, Retorno, Win Rate, Profit Fct, Max DD, Fees)
- Grid 2 col: tabla por Trigger / tabla por Sesión
- `#equity-container` — TradingView chart del equity curve (140px alto)
- Botones `.btn-start` / `.btn-stop` para iniciar/detener bot
- Tabla histórico trades (30 últimos)

Data: `/api/bot/metrics` + `/api/bot/trades`

#### G.6 `#rtab-sistema` (tab Sistema) — el más largo, 12 secciones

Cada sección es un `.sys-section` con `.sys-title`:
1. **Sentimiento**: meta FinBERT + narrativas + top articles + tabla por source
2. **Calendario Macro**: tabla eventos económicos
3. **On-Chain**: tabla métricas CryptoQuant
4. **Flujos ETF Spot BTC**: resumen 5d + tabla vehicles × dates
5. **Microstructure Futuros**: grid items (Funding, OI, Liq longs/shorts, Put/Call, etc.)
6. **Modelo ML Ensemble**: AUC + folds badges + top features con bars
7. **Confluence Score Pesos**: 6 weights con bars
8. **Bias Snapshot**: score grande + módulos + top drivers
9. **Estado del Pipeline**: lista key-value de scrapers
10. 🤖 **Agente IA — Metadata** *(nuevo)*: modelo, temp, tools por categoría
11. 📚 **RAG (FAISS + MiniLM)** *(nuevo)*: docs indexados, tamaño, build date
12. 📏 **RAGAS Evaluation** *(nuevo)*: 3 barras coloreadas + tabla por categoría

Data: `/api/system/transparency` + `/api/agent/info`

#### G.7 `#rtab-noticias` (tab Noticias)

- `#noticias-meta` — línea con stats
- `#noticias-stats` — grid 5 cards (Artículos 24h, Fuentes, FinBERT, Pos, Neg)
- `#noticias-narr` — grid 5 narratives pills (Inst/Macro/Reg/OnChain/Tech)
- `#noticias-filter-bar` — pills Todos/Positivos/Negativos
- `#noticias-grid` — grid auto-fill articles cards (min 290px)

Cada article card (`.news-full-card`):
- Border-left 3px verde/rojo/gris según sentiment
- Header con score pill + tier badge + source + tiempo
- Texto preview
- Decay bar (influence)
- Score raw + decay al pie

Data: `/api/system/transparency.top_articles + .narratives + .by_source`

#### G.8 `#rtab-educacion` (tab Educación) — nuevo, estático

- Header con título y subtítulo
- TOC con 10 anchor links (grid 2 columnas)
- 10 secciones `.edu-card` con border-left según tipo (up/down/neutral/warning):
  1. ¿Qué es ICT?
  2. Market Structure (con SVG diagrams)
  3. Order Blocks (ejemplo BTC $62k)
  4. Fair Value Gaps (ASCII art)
  5. Break of Structure
  6. Change of Character
  7. Liquidity Sweeps (mecha de sweep ASCII)
  8. Killzones
  9. Premium / Discount
  10. Confluence Score (tabla módulos)
- Footer

Cada card: `<h3>` con icono SVG + `<p>` explicación + ejemplo + `.edu-callout` "💡 Por qué importa".

---

## 🐛 Bugs/issues UX a arreglar

1. **Tab nav overflow**: con 8 tabs, en pantallas <1280px pueden saltar de línea o necesitan scroll horizontal. Verificar.
2. **Confluence bars**: el `--center-line` puede no estar bien centrado en todos los anchos.
3. **Position card**: en estado "no trade" el border-left del card está subtle, pero hay overflow visual con el label arriba.
4. **Agente output**: el HTML streaming a veces deja `<h3>` cortado mientras llega más texto. Cosmético.
5. **Sistema tab**: la tabla ETF puede salirse del card si hay muchos vehicles + dates.
6. **Noticias**: las cards a veces tienen alturas muy distintas (texto largo vs corto) → grid irregular.
7. **Educación**: el `<svg viewBox>` de Market Structure es bonito pero a veces no se renderiza con el color del trend correcto en todas las pantallas.
8. **Mobile (no soportado oficialmente)**: el layout se rompe en <800px porque el grid es fijo 1fr+680px.

---

## 📐 Mejoras visuales sugeridas (priorizadas)

### Prioridad ALTA (impacto visual grande)
- **Mejor jerarquía tipográfica**: el dashboard usa font-mono masivamente. Mezclar más sans-serif en cards educativas y narrativa.
- **Animaciones sutiles**: transiciones al cambiar valores numéricos (CSS `@property` con custom counters o JS framer-motion-like).
- **Skeleton loaders**: mientras cargan datos, mostrar shimmer placeholders en vez de "—" o vacío.
- **Empty states ilustrados**: cuando no hay datos (e.g., bot detenido, sin alertas), mostrar SVG ilustración + CTA.

### Prioridad MEDIA
- **Tooltips informativos**: hover sobre cada métrica → tooltip con explicación + benchmark.
- **Modal de detalle**: click en cualquier card → modal full con detalle expandido.
- **Tema light/dark toggle**: actualmente solo dark, añadir light.
- **Settings panel**: cog icon en topbar → modal con preferencias (refresh interval, hide modules, etc.).

### Prioridad BAJA
- **Achievements/gamification**: badges cuando bot alcanza milestones.
- **Audio alerts**: opcional, sonido al cambiar score significativamente.
- **Exportar a PDF**: botón en cada tab para snapshot.

---

## 🔌 API Endpoints disponibles (backend Flask)

| Endpoint | Refresh | Devuelve |
|---|---|---|
| `GET /api/live` | 5s | precio, change, high/low, vol, spread, source |
| `GET /api/session` | 30s | current_session, is_killzone, etc. |
| `GET /api/market` | manual | OHLCV + indicators por TF (1h, 4h) |
| `GET /api/chart/<tf>` | 60s | candles + ICT overlays |
| `GET /api/ict` | manual | ict_context por TF |
| `GET /api/levels` | manual | niveles técnicos con R:R |
| `GET /api/signals/breakdown` | 60s | confluence score + módulos + weights |
| `GET /api/sentiment` | manual | FinBERT diarios + agregados |
| `GET /api/fear_greed` | manual | F&G actual + history |
| `GET /api/bot/status` | 15s | is_running, capital, open_positions |
| `GET /api/bot/metrics` | manual | métricas trading + equity curve |
| `GET /api/bot/trades?limit=N` | manual | últimos N trades |
| `GET /api/bot/alerts` | 30s | alertas dedup top 12 |
| `POST /api/bot/start` | — | inicia bot |
| `POST /api/bot/stop` | — | detiene bot |
| `GET /api/agent/stream` | SSE | agent response streaming |
| `GET /api/agent/info` | manual | metadata agente + RAG + RAGAS |
| `GET /api/system/transparency` | manual | TODO: sentiment+macro+onchain+etf+micro+ml+weights+bias+pipeline |

---

## 🎯 Lo que necesitas pedirle a Claude Code mañana

```
Quiero un rediseño visual profesional del dashboard de trading BTC.
Stack: HTML/CSS/JS vanilla, TradingView Lightweight Charts.
Aesthetic: Bloomberg Terminal + Fintech moderno glassmorphism cyan.

Tengo TODOS los componentes documentados en DASHBOARD_COMPONENTS.md.

CONSTRAINTS DUROS:
- NO romper IDs ni data-* attributes (JS depende)
- NO tocar lógica JS (refreshX, fetch endpoints)
- NO añadir frameworks externos
- Mantener el design system de CSS variables en :root

OBJETIVOS:
1. Mejorar jerarquía visual y spacing
2. Animaciones sutiles en cambios de valores
3. Skeleton loaders en lugar de "—" vacíos
4. Empty states ilustrados
5. Tooltips informativos en métricas clave
6. Mejor responsive <1280px

Empieza por: [topbar / charts / quick stats / sistema tab — el que prefieras].

El archivo principal es agente_ia/templates/dashboard.html (~2000 líneas single-file).
```

---

**Fecha de generación**: 2026-06-07
**Versión del dashboard**: post-auditoría RAGAS + AUC honesto
**Última verificación funcional**: todos los endpoints OK, bot vivo PID 57070
