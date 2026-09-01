import { ChevronRight, CloudSun, Sun } from 'lucide-react'
import RangeBar from './RangeBar'

function Uncertainty({ uncertainty }) {
  const level = uncertainty === 'Low' ? 4 : 3
  return <span className="confidence" aria-label={`${uncertainty} forecast uncertainty`}><i className="confidence-bars" aria-hidden="true">{[0, 1, 2, 3].map((bar) => <b key={bar} className={bar < level ? 'on' : ''} />)}</i><span>{uncertainty}</span></span>
}

function ForecastSkeleton() {
  return <div className="forecast-skeleton" aria-label="Loading forecast"><i /><i /><i /><i /><i /></div>
}

export default function ForecastTable({ forecasts, loading, showBaseline, selectedStation, onSelectStation }) {
  if (loading && forecasts.length === 0) return <ForecastSkeleton />
  if (forecasts.length === 0) return <section className="empty-state"><CloudSun /><h2>Forecasts are temporarily unavailable</h2><p>Try refreshing in a moment. Your saved preferences are unchanged.</p></section>
  return <section className={showBaseline ? 'forecast-table show-baseline' : 'forecast-table'} aria-label="Daily high forecasts">
    <div className="forecast-table-head"><span>Station</span><span>Predicted high</span><span title="A fixed ±2°F planning band; historical coverage is shown in the forecast brief.">4° planning range</span>{showBaseline && <span title="Raw model correction compared with the NBM baseline">NBM / raw delta</span>}<span>Uncertainty</span><span className="sr-only">Details</span></div>
    {forecasts.map((forecast) => <article className={selectedStation === forecast.station ? 'forecast-row selected' : 'forecast-row'} key={forecast.station}>
      <div className="place"><span className="station">{forecast.station}</span><span>{forecast.city}</span></div>
      <div className="forecast-high"><Sun aria-hidden="true" /><strong>{forecast.highF}<sup>°F</sup></strong>{forecast.observedHighSoFarF !== null && forecast.observedHighSoFarF !== undefined && <em>Observed high so far</em>}</div>
      <RangeBar low={forecast.fourDegreeRangeLowF} high={forecast.fourDegreeRangeHighF} />
      {showBaseline && <div className="baseline-value"><strong>{forecast.baselineHighF}°</strong><span className={forecast.modelDeltaF >= 0 ? 'positive' : 'negative'}>{forecast.modelDeltaF > 0 ? '+' : ''}{forecast.modelDeltaF}°</span></div>}
      <Uncertainty uncertainty={forecast.uncertainty} />
      <button className="row-action" type="button" aria-label={`Show details for ${forecast.city}`} aria-pressed={selectedStation === forecast.station} onClick={() => onSelectStation(forecast.station)}><ChevronRight /></button>
    </article>)}
  </section>
}
