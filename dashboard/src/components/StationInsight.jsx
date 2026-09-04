import { Activity, ArrowUpRight, Gauge, Info, Radio, ThermometerSun } from 'lucide-react'

function signed(value) { return `${value > 0 ? '+' : ''}${value}°` }

export default function StationInsight({ forecast, accuracy, evidence, rankedCount }) {
  if (!forecast) return <aside className="insight-card empty-insight"><Info /><h2>Choose a station</h2><p>Select a row to see the model’s current estimate, baseline comparison, and historical error profile.</p></aside>
  const agreement = Number.isFinite(forecast.sourceAgreement) ? Math.round(forecast.sourceAgreement * 100) : null
  const isNwsFallback = !forecast.isCalibrated
  const hasBand = Number.isFinite(forecast.rangeLowF) && Number.isFinite(forecast.rangeHighF)
  const sourceState = forecast.dataFreshness >= 1 ? 'Fresh source response' : 'Source response may be stale'
  return <aside className="insight-card" aria-labelledby="insight-title">
    <div className="insight-heading"><div><span>{forecast.station}</span><h2 id="insight-title">{forecast.city}</h2></div>{accuracy ? <span className="accuracy-rank">#{accuracy.rank} historical MAE</span> : <span className="accuracy-rank accuracy-rank-muted">Not yet calibrated</span>}</div>
    <div className="insight-primary"><ThermometerSun /><div><span>Predicted high</span><strong>{forecast.highF}°<sup>F</sup></strong>{forecast.observedHighSoFarF !== null && <small>Observed high so far: {forecast.observedHighSoFarF}°F</small>}</div></div>
    <div className="insight-grid"><div><span>Fixed ±2°F planning reference</span><strong>{Number.isFinite(forecast.fourDegreeRangeLowF) ? `${forecast.fourDegreeRangeLowF}°–${forecast.fourDegreeRangeHighF}°` : 'Unavailable'}</strong></div><div><span>80% calibrated interval</span><strong>{hasBand ? `${forecast.rangeLowF}°–${forecast.rangeHighF}°` : 'Unavailable'}</strong></div><div><span>{isNwsFallback ? 'NWS forecast high' : 'NBM baseline'}</span><strong>{forecast.baselineHighF}°F</strong></div><div><span>{isNwsFallback ? 'Forecast source' : 'Raw model vs NBM'}</span><strong>{isNwsFallback ? 'NWS' : `${forecast.rawModelHighF}°F · ${signed(forecast.modelDeltaF)}`}</strong></div></div>
    <div className="insight-section evidence-section"><div className="insight-section-heading"><Radio size={16} /><h3>Live model evidence</h3><span className={forecast.dataFreshness >= 1 ? 'evidence-status fresh' : 'evidence-status stale'}>{sourceState}</span></div><div className="agreement-row"><div><span>Source agreement</span><strong>{agreement == null ? '—' : `${agreement}%`}</strong></div><div className="agreement-track" aria-label={agreement == null ? 'Source agreement unavailable' : `Source agreement ${agreement}%`}><i style={{ width: `${agreement ?? 0}%` }} /></div></div><p className="evidence-caption">Agreement compares the available source guidance. It describes model spread, not the chance this forecast settles correctly.</p></div>
    {accuracy && !isNwsFallback ? <div className="insight-section"><div className="insight-section-heading"><Activity size={16} /><h3>Historical accuracy</h3></div><dl><div><dt>Average error</dt><dd>{accuracy.maeF}°F</dd></div><div><dt>Within 2°F</dt><dd>{accuracy.within2Pct}%</dd></div><div><dt>90th-percentile error</dt><dd>{accuracy.p90ErrorF}°F</dd></div><div><dt>Average bias</dt><dd>{signed(accuracy.biasF)}</dd></div></dl></div>
      : <div className="insight-section"><div className="insight-section-heading"><Activity size={16} /><h3>Historical accuracy</h3></div><p className="insight-uncalibrated">This settlement station is configured but not yet in the trained model set. It shows a live NBM baseline with a NO BET status instead of an unvalidated residual correction.</p></div>}
    {forecast.reasonCodes?.length > 0 && <div className="insight-section insight-reasons"><div className="insight-section-heading"><Info size={16} /><h3>Why this estimate</h3></div><ul>{forecast.reasonCodes.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>}
    {accuracy && !isNwsFallback && <p className="insight-note"><Gauge size={15} />This station ranks {accuracy.rank} of {rankedCount ?? accuracy.rank} for historical average error. Overall, the candidate averages {evidence?.candidateMaeF ?? '—'}°F error across the evaluated forecast history.</p>}
    <a className="text-link" href="#/accuracy">How we measure performance <ArrowUpRight size={15} /></a>
  </aside>
}
