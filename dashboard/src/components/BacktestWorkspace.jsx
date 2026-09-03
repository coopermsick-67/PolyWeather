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

const round = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—'
const percent = (value) => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—'

function dateLabel(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function BacktestChart({ rows, candidateKey }) {
  if (!rows?.length || !candidateKey) return <div className="backtest-empty">Run a backtest to inspect forecast error by date.</div>
  const field = { xgb: 'xgbF', consensus: 'consensusF', blend: 'blendF', ridge: 'ridgeF', nbm: 'nbmF', hrrr: 'hrrrF', gfs: 'gfsF' }[candidateKey]
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
  return <div className="backtest-chart-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Backtest forecast compared with observed daily high">
    {ticks.map((tick) => {
      const y = top + (1 - (tick - low) / Math.max(high - low, 1)) * (height - top - bottom)
      return <g key={tick}><line x1={left} x2={width - right} y1={y} y2={y} className="backtest-grid" /><text x="2" y={y + 4} className="backtest-axis">{Math.round(tick)}°</text></g>
    })}
    {showBand && <polygon points={band} className="backtest-band" />}
    <polyline points={line('observedF')} className="backtest-observed" />
    <polyline points={line(field)} className="backtest-candidate" />
    {labels.map((row) => {
      const index = rows.indexOf(row)
      const x = left + (index / Math.max(rows.length - 1, 1)) * (width - left - right)
      return <text key={row.date} x={x} y={height - 8} textAnchor="middle" className="backtest-axis">{row.date.slice(5)}</text>
    })}
  </svg></div>
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
  const [status, setStatus] = useState({ state: 'loading', progress: 0, message: 'Loading the validated historical forecast archive.' })
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
        setStatus({ state: 'ready', progress: 0, message: 'Choose a city and run the historical evaluation.' })
      })
      .catch((error) => active && setStatus({ state: 'error', progress: 0, message: error.message }))
    return () => { active = false; activeRunId.current = null; workerRef.current?.terminate() }
  }, [])

  const stations = useMemo(() => registry.filter((item) => payload?.stations?.includes(item.stationId)), [registry, payload])
  const selected = result?.summaries?.find((summary) => summary.key === result.candidateKey)
  const dateRangeInvalid = Boolean(startDate && endDate && startDate > endDate)
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
      if (data.type === 'complete') { setResult(data.result); setStatus({ state: 'complete', progress: 1, message: `Verified: ${data.result.rows.toLocaleString()} of ${data.result.archiveRows.toLocaleString()} archive rows scored in ${data.result.executionMs}ms · ${data.result.archiveFingerprint}.` }); worker.terminate() }
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
      <div className="backtest-control"><span>Settlement city</span><LocationPicker stations={stations} value={station === 'all' ? null : station} onChange={(value) => setStation(value ?? 'all')} allowAll allLabel="All validated cities" /></div>
      <label className="backtest-control"><span>Start date</span><input type="date" value={startDate} min={payload?.dateRange.start} max={endDate || payload?.dateRange.end} onChange={(event) => setStartDate(event.target.value)} /></label>
      <label className="backtest-control"><span>End date</span><input type="date" value={endDate} min={startDate || payload?.dateRange.start} max={payload?.dateRange.end} onChange={(event) => setEndDate(event.target.value)} /></label>
      <fieldset className="backtest-sources"><legend>Forecasts to score</legend><div>{MODEL_OPTIONS.map(([key, label]) => <label key={key}><input type="checkbox" checked={models.includes(key)} onChange={() => toggleModel(key)} /><span>{label}</span></label>)}</div><small>Inputs are archived NBM, HRRR, and GFS guidance. NOAA/NWS is live-only because no immutable NWS forecast archive is available here.</small></fieldset>
      {dateRangeInvalid && <p className="backtest-date-error" role="alert">Start date must be on or before the end date.</p>}
      <button className="button backtest-run" type="button" onClick={run} disabled={status.state === 'loading' || status.state === 'starting' || status.state === 'running' || !models.length || dateRangeInvalid}><Play size={16} />{status.state === 'starting' || status.state === 'running' ? 'Running…' : 'Run backtest'}</button>
    </section>
    <section className={`backtest-progress ${status.state}`} aria-live="polite"><div>{status.state === 'starting' || status.state === 'running' ? <RotateCw className="spin" size={19} /> : status.state === 'complete' ? <CheckCircle2 size={19} /> : status.state === 'error' ? <XCircle size={19} /> : <Clock3 size={19} />}<strong>{status.state === 'starting' ? 'Waiting for worker acknowledgment' : status.state === 'running' ? 'Computing held-out scores' : status.state === 'complete' ? 'Verified backtest result' : status.state === 'error' ? 'Backtest did not run' : 'Ready to evaluate'}</strong><span>{status.message}</span></div>{(status.state === 'starting' || status.state === 'running') && <div className="backtest-progress-track"><i style={{ width: `${Math.round(status.progress * 100)}%` }} /></div>}<small>Results appear only after the worker has scored matching immutable rows. Runs in this tab; closing or reloading ends the run.</small></section>
    {result && <><section className="backtest-metrics"><Metric label="Mean absolute error" value={`${round(selected?.mae)}°F`} note={`${selected?.n ?? 0} held-out station-days`} /><Metric label="Bias" value={`${selected?.bias > 0 ? '+' : ''}${round(selected?.bias)}°F`} note="Positive means forecast warmer" tone="amber" /><Metric label="Within 2°F" value={percent(selected?.within2)} note="Share of scored forecasts" tone="green" /><Metric label="90% interval coverage" value={selected?.key === 'xgb' ? percent(selected?.intervalCoverage) : '—'} note={selected?.key === 'xgb' ? 'Observed within archived P10–P90' : 'Available for residual MOS'} /></section>
      <section className="backtest-results"><article className="backtest-chart-panel"><div className="backtest-panel-heading"><div><h2>Forecast vs. observed</h2><p>{result.dateRange.start} to {result.dateRange.end} · daily high °F</p></div><div className="backtest-legend"><span><i className="observed" />Observed NCEI TMAX</span><span><i className="candidate" />Selected forecast</span>{result.candidateKey === 'xgb' && <span><i className="band" />Archived P10–P90</span>}</div></div><BacktestChart rows={result.chartRows} candidateKey={result.candidateKey} /></article><article className="backtest-score-panel"><h2>Model comparison</h2><div className="backtest-score-list">{result.summaries.map((summary) => <div key={summary.key}><strong>{summary.label}</strong><span>MAE {round(summary.mae)}°F</span><span>{percent(summary.within2)} within 2°F</span><span>{percent(summary.coverage)} coverage</span></div>)}</div></article></section>
      <section className="backtest-table-panel"><div><h2>Scored rows</h2><span>{result.rows} matching held-out forecasts</span></div><div className="backtest-table-scroll"><table><thead><tr><th>Date</th><th>Station</th><th>Observed</th><th>WeatherPicks</th><th>NBM</th><th>HRRR</th><th>GFS</th></tr></thead><tbody>{result.recentRows.map((row) => <tr key={`${row.station}-${row.date}`}><td>{dateLabel(row.date)}</td><td>{row.station}</td><td>{round(row.observedF, 1)}°</td><td>{round(row.xgbF, 1)}°</td><td>{round(row.nbmF, 1)}°</td><td>{round(row.hrrrF, 1)}°</td><td>{round(row.gfsF, 1)}°</td></tr>)}</tbody></table></div></section>
      <aside className="backtest-disclosure"><AlertTriangle size={19} /><div><strong>What this does—and does not—show</strong><p>{result.contract} {result.limitation} {result.liveParity}</p></div></aside>
    </>}
  </div></main>
}
