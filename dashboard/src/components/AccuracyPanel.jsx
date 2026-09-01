import { BarChart3 } from 'lucide-react'

export default function AccuracyPanel({ accuracy, evidence, onSelectStation }) {
  if (!accuracy?.length) return null
  return <section className="accuracy-panel" aria-labelledby="accuracy-panel-title"><div className="accuracy-title"><div><BarChart3 /><div><h2 id="accuracy-panel-title">Accuracy by station</h2><p>Strict rolling backtest · lower error is better.</p></div></div><strong>{evidence?.candidateMaeF ?? '—'}°F overall</strong></div><ol>{accuracy.map((station) => <li key={station.station}><button type="button" onClick={() => onSelectStation(station.station)}><span className="rank-number">{station.rank}</span><span className="accuracy-station"><b>{station.station}</b><small>{station.within2Pct}% within 2°F</small></span><strong>{station.maeF}°F</strong></button></li>)}</ol><p className="accuracy-foot">Candidate improves on raw NBM by {evidence?.skillPct ?? '—'}% MAE skill across {evidence?.testForecasts?.toLocaleString() ?? '—'} held-out forecasts.</p></section>
}
