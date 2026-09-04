import { BarChart3 } from 'lucide-react'

export default function AccuracyPanel({ accuracy, evidence, onSelectStation, selectedStation }) {
  if (!accuracy?.length) return null
  return <section className="accuracy-panel" aria-labelledby="accuracy-panel-title">
    <div className="accuracy-title">
      <div><BarChart3 /><div><h2 id="accuracy-panel-title">City accuracy leaderboard</h2><p>Archived-composite evaluation · lower error is better.</p></div></div>
      <strong>{evidence?.candidateMaeF ?? '—'}°F overall MAE</strong>
    </div>
    <div className="leaderboard-scroll">
      <table className="leaderboard-table">
        <thead>
          <tr><th scope="col">Rank</th><th scope="col">Station</th><th scope="col">MAE (°F)</th><th scope="col">P90 error (°F)</th><th scope="col">Bias</th><th scope="col">Within 2°F</th></tr>
        </thead>
        <tbody>
          {accuracy.map((station) => (
            <tr key={station.station} className={station.station === selectedStation ? 'selected' : ''} onClick={() => onSelectStation(station.station)} tabIndex={0} role="button" aria-label={`View ${station.station}`} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelectStation(station.station) } }}>
              <td><span className={`leaderboard-rank ${station.rank <= 3 ? 'top' : ''}`}>{station.rank}</span></td>
              <td className="leaderboard-station">{station.station}</td>
              <td>{station.maeF.toFixed(2)}</td>
              <td>{station.p90ErrorF.toFixed(2)}</td>
              <td className={station.biasF > 0 ? 'positive' : station.biasF < 0 ? 'negative' : ''}>{station.biasF > 0 ? '+' : ''}{station.biasF.toFixed(2)}</td>
              <td>
                <div className="leaderboard-hitrate">
                  <div className="leaderboard-track" aria-hidden="true"><i style={{ width: `${Math.min(100, station.within2Pct)}%` }} /></div>
                  <span>{station.within2Pct}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    <p className="accuracy-foot">Candidate improves on raw NBM by {evidence?.skillPct ?? '—'}% MAE skill across {evidence?.testForecasts?.toLocaleString() ?? '—'} held-out archived forecasts. &ldquo;Within 2&deg;F&rdquo; is a historical accuracy rate, not a measured live-refresh win rate or a market hit rate.</p>
  </section>
}
