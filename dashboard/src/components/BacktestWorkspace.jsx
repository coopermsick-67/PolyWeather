import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Clock3, Play, RotateCw, ShieldAlert, XCircle } from 'lucide-react'
import LocationPicker from './LocationPicker'

const MODEL_OPTIONS = [
  ['xgb', 'WeatherPicks residual MOS'],
  ['consensus', 'NBM/HRRR/GFS consensus'],
  ['blend', 'WeatherPicks blended benchmark'],
  ['ridge', 'WeatherPicks ridge benchmark'],
  ['nbm', 'NCEP NBM'],
  ['hrrr', 'HRRR'],
  ['gfs', 'GFS'],
]

const MODEL_FIELDS = Object.fromEntries(MODEL_OPTIONS.map(([key]) => [key, `${key}F`]))
MODEL_FIELDS.xgb = 'xgbF'
MODEL_FIELDS.consensus = 'consensusF'
const MODEL_LABELS = Object.fromEntries(MODEL_OPTIONS)

const round = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—'
const percent = (value) => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—'

function dateLabel(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function BacktestChart({ rows, candidateKey }) {
  if (!rows?.length || !candidateKey) return <div className="backtest-empty">Run a backtest to inspect forecast error by date.</div>
  const field = MODEL_FIELDS[candidateKey]
  const showBand = candidateKey === 'xgb' && rows.some((row) => Number.isFinite(row.p10F) && Number.isFinite(row.p90F))
  const values = rows.flatMap((row) => [row.observedF, row[field], ...(showBand ? [row.p10F, row.p90F] : [])]).filter(Number.isFinite)
  const low = Math.floor(Math.min(...values) / 5) * 5
  const high = Math.ceil(Math.max(...values) / 5) * 5
  const width = 860
  const height = 250
  const left = 36
  const right = 12
  const top = 12
  const bottom = 34
  const point = (value, index) => {
    const x = left + (index / Math.max(rows.length - 1, 1)) * (width - left - right)
    const y = top + (1 - (value - low) / Math.max(high - low, 1)) * (height - top - bottom)
    return `${x},${y}`
  }
  const line = (fieldName) => rows.map((row, index) => point(row[fieldName], index)).join(' ')
  const band = `${rows.map((row, index) => point(row.p90F, index)).join(' ')} ${rows.slice().reverse().map((row, reverseIndex) => point(row.p10F, rows.length - reverseIndex - 1)).join(' ')}`
  const ticks = Array.from({ length: 5 }, (_, index) => low + ((high - low) * index) / 4)
  const labels = rows.filter((_, index) => index % Math.max(1, Math.floor(rows.length / 6)) === 0)
  return <figure className="backtest-chart-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="backtest-chart-title backtest-chart-desc">
    <title id="backtest-chart-title">Archived forecast compared with observed daily high</title>
    <desc id="backtest-chart-desc">One station timeline from the selected historical rows. The black line is official observed NCEI daily maximum temperature and the blue line is the selected archived forecast.</desc>
    {ticks.map((tick) => {
      const y = top + (1 - (tick - low) / Math.max(high - low, 1)) * (height - top - bottom)
      return <g key={tick}><line x1={left} x2={width - right} y1={y} y2={y} className="backtest-grid" /><text x="2" y={y + 4} className="backtest-axis">{Math.round(tick)}°</text></g>
    })}
    {showBand && <polygon points={band} className="backtest-band" />}
    <polyline points={line('observedF')} className="backtest-observed" />
    <polyline points={line(field)} className="backtest-candidate" />
    {rows.map((row, index) => <circle key={row.date} cx={left + (index / Math.max(rows.length - 1, 1)) * (width - left - right)} cy={top + (1 - (row[field] - low) / Math.max(high - low, 1)) * (height - top - bottom)} r="2.2" className="backtest-candidate-point" />)}
    {labels.map((row) => {
      const index = rows.indexOf(row)
      const x = left + (index / Math.max(rows.length - 1, 1)) * (width - left - right)
      return <text key={row.date} x={x} y={height - 8} textAnchor="middle" className="backtest-axis">{row.date.slice(5)}</text>
    })}
  </svg><figcaption>Each point is an archived forecast paired with its official observed local-day high. The timeline is sampled for legibility; aggregate scores use every matching archived row.</figcaption></figure>
}

function BacktestErrorChart({ rows, candidateKey }) {
  const field = MODEL_FIELDS[candidateKey]
  const points = (rows ?? []).filter((row) => Number.isFinite(row.observedF) && Number.isFinite(row[field]))
  if (!points.length) return null
  const width = 860
  const height = 106
  const left = 36
  const right = 12
  const top = 14
  const bottom = 24
  const maxError = Math.max(2, Math.ceil(Math.max(...points.map((row) => Math.abs(row[field] - row.observedF))) / 2) * 2)
  const x = (index) => left + (index / Math.max(points.length - 1, 1)) * (width - left - right)
  const y = (error) => top + (1 - (error + maxError) / (maxError * 2)) * (height - top - bottom)
  const baseline = y(0)
  const labels = points.filter((_, index) => index % Math.max(1, Math.floor(points.length / 5)) === 0)
  return <figure className="backtest-error-chart"><div><strong>Error by date</strong><span>Above zero = forecast warmer than observed</span></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Forecast error by date">
    <line x1={left} x2={width - right} y1={baseline} y2={baseline} className="backtest-error-baseline" />
    <text x="2" y={top + 4} className="backtest-axis">+{maxError}°</text><text x="2" y={baseline + 4} className="backtest-axis">0°</text><text x="2" y={height - bottom + 4} className="backtest-axis">−{maxError}°</text>
    {points.map((row, index) => { const error = row[field] - row.observedF; return <g key={row.date}><line x1={x(index)} x2={x(index)} y1={baseline} y2={y(error)} className={error >= 0 ? 'backtest-error-warm' : 'backtest-error-cool'} /><circle cx={x(index)} cy={y(error)} r="2.3" className={error >= 0 ? 'backtest-error-warm' : 'backtest-error-cool'} /></g> })}
    {labels.map((row) => { const index = points.indexOf(row); return <text key={row.date} x={x(index)} y={height - 4} textAnchor="middle" className="backtest-axis">{row.date.slice(5)}</text> })}
  </svg></figure>
}

function Metric({ label, value, note, tone = '' }) {
  return <article className={`backtest-metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>
}

export default function BacktestWorkspace({ registry = [] }) {
  const workerRef = useRef(null)
  const activeRunId = useRef(null)
  const [payload, setPayload] = useState(null)
  const [station, setStation] = useState('all')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [models, setModels] = useState(['xgb', 'consensus', 'nbm', 'hrrr', 'gfs', 'blend'])
  const [status, setStatus] = useState({ state: 'loading', progress: 0, message: 'Loading the historical archived-composite evaluation data.' })
  const [result, setResult] = useState(null)

  useEffect(() => {
    let active = true
    fetch('/backtest-data.json', { cache: 'no-store' })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Backtest archive is unavailable.')))
      .then((next) => {
        if (!Array.isArray(next.rows) || !next.rows.length || !next.dateRange?.start || !next.dateRange?.end || !next.fingerprint) throw new Error('Backtest archive failed integrity checks.')
        if (!active) return
        setPayload(next)
        setStartDate(next.dateRange.start)
        setEndDate(next.dateRange.end)
        setStatus({ state: 'ready', progress: 0, message: 'Choose a city and score the archived historical evaluation.' })
      })
      .catch((error) => active && setStatus({ state: 'error', progress: 0, message: error.message }))
    return () => { active = false; activeRunId.current = null; workerRef.current?.terminate() }
  }, [])

  // This must not depend on the live /api/dashboard fetch (registry) finishing
  // first: that request can take 15-20s, while backtest-data.json is a small
  // static file that's typically ready almost immediately. Build the list from
  // the archive's own station list (payload.stations) as soon as it loads, and
  // layer in nicer display names from registry only once/if it has arrived --
  // never let a slow, unrelated fetch leave this picker showing zero cities.
  const stations = useMemo(() => {
    if (!payload?.stations) return []
    const byId = new Map(registry.map((item) => [item.stationId, item]))
    return payload.stations.map((stationId) => byId.get(stationId) ?? { stationId, display_name: stationId, name: '' })
  }, [registry, payload])
  const selected = result?.summaries?.find((summary) => summary.key === result.candidateKey)
  const dateRangeInvalid = Boolean(startDate && endDate && startDate > endDate)
  const scopeRows = useMemo(() => payload?.rows?.filter((row) => (station === 'all' || row.station === station) && row.date >= startDate && row.date <= endDate).length ?? 0, [payload, station, startDate, endDate])
  const selectedModelLabel = result ? (MODEL_LABELS[result.candidateKey] ?? 'Selected forecast') : ''
  const rankedSummaries = result ? [...result.summaries].sort((left, right) => (left.mae ?? Infinity) - (right.mae ?? Infinity)) : []
  const run = () => {
    if (!payload || !models.length || dateRangeInvalid) return
    workerRef.current?.terminate()
    const runId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
    activeRunId.current = runId
    const worker = new Worker('/backtest-worker.js')
    workerRef.current = worker
    setStatus({ state: 'starting', progress: 0, message: 'Creating a dedicated scoring worker.' })
    worker.onmessage = ({ data }) => {
      if (data.runId !== activeRunId.current) return
      if (data.type === 'started') setStatus({ state: 'running', progress: 0.05, message: data.message })
      if (data.type === 'progress') setStatus({ state: 'running', progress: data.progress, message: data.message })
      if (data.type === 'complete') { setResult(data.result); setStatus({ state: 'complete', progress: 1, message: `Complete: ${data.result.rows.toLocaleString()} of ${data.result.archiveRows.toLocaleString()} archived rows scored in ${data.result.executionMs}ms · ${data.result.archiveFingerprint}.` }); worker.terminate() }
      if (data.type === 'error') { setStatus({ state: 'error', progress: 0, message: data.message }); worker.terminate() }
    }
    worker.onerror = () => { if (activeRunId.current === runId) setStatus({ state: 'error', progress: 0, message: 'The backtest worker failed before producing results. Nothing was scored.' }); worker.terminate() }
    worker.onmessageerror = () => { if (activeRunId.current === runId) setStatus({ state: 'error', progress: 0, message: 'The backtest worker returned unreadable data. Nothing was scored.' }); worker.terminate() }
    worker.postMessage({ type: 'run', runId, payload, station, startDate, endDate, models })
  }
  const toggleModel = (key) => setModels((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key])

  return <main className="backtest-page"><div className="page-width">
    <div className="backtest-heading"><div><p className="section-label">HISTORICAL EVALUATION</p><h1>Run a backtest</h1><p>Score the same held-out station-days used for model evaluation. This is an audit tool, not a replay of a new forecast run.</p></div><div className="backtest-contract"><ShieldAlert size={17} /><span>{payload?.decision === 'SHADOW_ONLY' ? 'Shadow-only model' : 'Historical evaluation'}</span></div></div>
    <section className="backtest-controls" aria-label="Backtest controls">
      <div className="backtest-control"><span>Settlement city</span><LocationPicker stations={stations} value={station === 'all' ? null : station} onChange={(value) => setStation(value ?? 'all')} allowAll allLabel="All evaluated cities" /></div>
      <label className="backtest-control"><span>Start date</span><input type="date" value={startDate} min={payload?.dateRange.start} max={endDate || payload?.dateRange.end} onChange={(event) => setStartDate(event.target.value)} /></label>
      <label className="backtest-control"><span>End date</span><input type="date" value={endDate} min={startDate || payload?.dateRange.start} max={payload?.dateRange.end} onChange={(event) => setEndDate(event.target.value)} /></label>
      <fieldset className="backtest-sources"><legend>Forecasts to score</legend><div>{MODEL_OPTIONS.map(([key, label]) => <label key={key}><input type="checkbox" checked={models.includes(key)} onChange={() => toggleModel(key)} /><span>{label}</span></label>)}</div><small>Archived inputs include NBM, HRRR, and GFS guidance. NOAA/NWS is intentionally excluded because no immutable NWS forecast archive is available.</small></fieldset>
      {dateRangeInvalid && <p className="backtest-date-error" role="alert">Start date must be on or before the end date.</p>}
      <button className="button backtest-run" type="button" onClick={run} disabled={status.state === 'loading' || status.state === 'starting' || status.state === 'running' || !models.length || dateRangeInvalid}><Play size={16} />{status.state === 'starting' || status.state === 'running' ? 'Running…' : 'Run backtest'}</button>
      <div className="backtest-scope" aria-live="polite"><span>Archive scope</span><strong>{scopeRows.toLocaleString()} held-out rows</strong><small>{station === 'all' ? 'All evaluated cities' : station} · {models.length} forecast{models.length === 1 ? '' : 's'} selected · immutable historical archive</small></div>
    </section>
    <section className={`backtest-progress ${status.state}`} aria-live="polite"><div>{status.state === 'starting' || status.state === 'running' ? <RotateCw className="spin" size={19} /> : status.state === 'complete' ? <CheckCircle2 size={19} /> : status.state === 'error' ? <XCircle size={19} /> : <Clock3 size={19} />}<strong>{status.state === 'starting' ? 'Waiting for worker acknowledgment' : status.state === 'running' ? 'Computing archived scores' : status.state === 'complete' ? 'Historical evaluation complete' : status.state === 'error' ? 'Backtest did not run' : 'Ready to evaluate'}</strong><span>{status.message}</span></div>{(status.state === 'starting' || status.state === 'running') && <div className="backtest-progress-track"><i style={{ width: `${Math.round(status.progress * 100)}%` }} /></div>}<small>Results appear only after the worker scores matching static archive rows. This tab closes the run when it reloads; it is not a prospective live-forecast replay.</small></section>
    {result && <><section className="backtest-metrics" aria-label={`${selectedModelLabel} evaluation metrics`}><Metric label="Mean absolute error" value={`${round(selected?.mae)}°F`} note={`${selected?.n ?? 0} held-out station-days`} /><Metric label="Bias" value={`${selected?.bias > 0 ? '+' : ''}${round(selected?.bias)}°F`} note="Positive means forecast warmer" tone="amber" /><Metric label="Within 2°F" value={percent(selected?.within2)} note="Share of scored forecasts" tone="green" /><Metric label="Archived interval coverage" value={selected?.key === 'xgb' ? percent(selected?.intervalCoverage) : '—'} note={selected?.key === 'xgb' ? 'Observed within the residual-MOS archive interval' : 'Available for residual MOS only'} /></section>
      <section className="backtest-results"><article className="backtest-chart-panel"><div className="backtest-panel-heading"><div><h2>Forecast vs. observed</h2><p>{result.chartStation} · {result.dateRange.start} to {result.dateRange.end} · daily high °F{station === 'all' ? ' · aggregate scores cover all selected cities' : ''}</p></div><div className="backtest-legend"><span><i className="observed" />Observed NCEI TMAX</span><span><i className="candidate" />{selectedModelLabel}</span>{result.candidateKey === 'xgb' && <span><i className="band" />Archived interval</span>}</div></div><BacktestChart rows={result.chartRows} candidateKey={result.candidateKey} /><BacktestErrorChart rows={result.chartRows} candidateKey={result.candidateKey} /></article><article className="backtest-score-panel"><div className="backtest-score-heading"><div><h2>Model comparison</h2><p>Same archived rows · ordered by MAE</p></div><span>{result.summaries.length} scored</span></div><div className="backtest-score-list">{rankedSummaries.map((summary, index) => <div key={summary.key} className={summary.key === result.candidateKey ? 'selected' : ''}><b aria-label={`Rank ${index + 1}`}>{index + 1}</b><strong>{summary.label}{summary.key === result.candidateKey && <em>Charted</em>}</strong><span>MAE {round(summary.mae)}°F</span><span>RMSE {round(summary.rmse)}°F</span><span>{percent(summary.within2)} within 2°F</span><span>{percent(summary.coverage)} coverage</span></div>)}</div></article></section>
      <section className="backtest-table-panel"><div><div><h2>Scored rows</h2><p>Latest 12 rows for the station shown in the timeline.</p></div><span>{result.rows.toLocaleString()} matching held-out archived forecasts</span></div><div className="backtest-table-scroll"><table><thead><tr><th>Date</th><th>Station</th><th>Observed</th><th>{selectedModelLabel}</th><th>Signed error</th><th>NBM</th><th>HRRR</th><th>GFS</th></tr></thead><tbody>{result.recentRows.map((row) => { const forecast = row[MODEL_FIELDS[result.candidateKey]]; const error = Number.isFinite(forecast) ? forecast - row.observedF : null; return <tr key={`${row.station}-${row.date}`}><td>{dateLabel(row.date)}</td><td>{row.station}</td><td>{round(row.observedF, 1)}°</td><td className="backtest-table-model">{round(forecast, 1)}°</td><td className={error === null ? '' : error > 0 ? 'backtest-error-positive' : error < 0 ? 'backtest-error-negative' : ''}>{error === null ? '—' : `${error > 0 ? '+' : ''}${round(error, 1)}°`}</td><td>{round(row.nbmF, 1)}°</td><td>{round(row.hrrrF, 1)}°</td><td>{round(row.gfsF, 1)}°</td></tr> })}</tbody></table></div></section>
      <aside className="backtest-disclosure"><AlertTriangle size={19} /><div><strong>What this does—and does not—show</strong><p>{result.contract} {result.limitation} {result.liveParity}</p></div></aside>
    </>}
  </div></main>
}
