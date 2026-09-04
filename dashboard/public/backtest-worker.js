const MODELS = {
  xgb: { label: 'WeatherPicks residual MOS', field: 'xgbF' },
  consensus: { label: 'NBM/HRRR/GFS weighted consensus', field: 'consensusF' },
  blend: { label: 'WeatherPicks blended benchmark', field: 'blendF' },
  ridge: { label: 'WeatherPicks ridge benchmark', field: 'ridgeF' },
  nbm: { label: 'NCEP NBM', field: 'nbmF' },
  hrrr: { label: 'HRRR', field: 'hrrrF' },
  gfs: { label: 'GFS', field: 'gfsF' },
}

const average = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
const finite = (value) => typeof value === 'number' && Number.isFinite(value)

function summarize(rows, field, label, key) {
  const usable = rows.filter((row) => finite(row.observedF) && finite(row[field]))
  const errors = usable.map((row) => row[field] - row.observedF)
  return {
    key: key ?? Object.entries(MODELS).find(([, model]) => model.field === field)?.[0] ?? field,
    label,
    n: usable.length,
    coverage: rows.length ? usable.length / rows.length : 0,
    mae: average(errors.map(Math.abs)),
    rmse: errors.length ? Math.sqrt(average(errors.map((error) => error ** 2))) : null,
    bias: average(errors),
    within2: errors.length ? errors.filter((error) => Math.abs(error) <= 2).length / errors.length : 0,
    intervalCoverage: key === 'xgb' && errors.length
      ? usable.filter((row) => finite(row.p10F) && finite(row.p90F) && row.observedF >= row.p10F && row.observedF <= row.p90F).length / usable.filter((row) => finite(row.p10F) && finite(row.p90F)).length
      : null,
  }
}

async function runBacktest({ payload, station, startDate, endDate, models, runId }) {
  const startedAt = performance.now()
  if (!payload?.rows?.length) throw new Error('The immutable backtest archive is empty or unavailable.')
  if (!startDate || !endDate || startDate > endDate) throw new Error('Choose a valid start and end date.')
  self.postMessage({ type: 'started', runId, message: 'Worker accepted the run and is validating the immutable archive.' })
  const sourceRows = payload.rows.filter((row) =>
    (station === 'all' || row.station === station)
    && row.date >= startDate
    && row.date <= endDate
  )
  if (!sourceRows.length) throw new Error('No held-out forecast rows match that city and date range.')
  self.postMessage({ type: 'progress', runId, progress: 0.2, message: 'Filtering immutable held-out forecast rows' })
  await new Promise((resolve) => setTimeout(resolve, 0))
  const selected = models.filter((key) => MODELS[key])
  if (!selected.length) throw new Error('Choose at least one forecast to score.')
  const summaries = []
  for (let index = 0; index < selected.length; index += 1) {
    const model = MODELS[selected[index]]
    summaries.push(summarize(sourceRows, model.field, model.label, selected[index]))
    self.postMessage({ type: 'progress', runId, progress: 0.2 + ((index + 1) / Math.max(selected.length, 1)) * 0.65, message: `Scoring ${model.label}` })
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  const candidate = summaries.find((summary) => summary.key === 'xgb') ?? summaries[0]
  // A line chart has one continuous time axis. Never interleave different
  // cities (their different climates create the misleading zig-zag shown in
  // the old all-city chart). Aggregate scores remain all-city; the visual
  // drill-down uses one deterministic station.
  const chartStation = station === 'all' ? [...new Set(sourceRows.map((row) => row.station))].sort()[0] : station
  const chartRows = sourceRows
    .filter((row) => row.station === chartStation)
    .filter((row) => candidate && finite(row.observedF) && finite(row[MODELS[candidate.key].field]))
    .sort((left, right) => left.date.localeCompare(right.date))
  const step = Math.max(1, Math.ceil(chartRows.length / 72))
  const displayRows = chartRows.filter((_, index) => index % step === 0 || index === chartRows.length - 1)
  self.postMessage({
    type: 'complete',
    runId,
    result: {
      rows: sourceRows.length,
      archiveRows: payload.rows.length,
      archiveFingerprint: payload.fingerprint ?? 'unfingerprinted',
      executionMs: Math.round(performance.now() - startedAt),
      runId: `${station}:${startDate}:${endDate}:${selected.join(',')}:${sourceRows.length}`,
      dateRange: { start: startDate, end: endDate },
      summaries,
      candidateKey: candidate?.key ?? null,
      chartStation,
      chartRows: displayRows,
      recentRows: chartRows.slice(-12).reverse(),
      decision: payload.decision,
      contract: payload.contract,
      limitation: payload.limitation,
      liveParity: payload.liveParity,
    },
  })
}

self.onmessage = (event) => {
  if (event.data?.type !== 'run') return
  runBacktest(event.data).catch((error) => self.postMessage({ type: 'error', runId: event.data.runId, message: error.message || 'Backtest could not be completed.' }))
}
