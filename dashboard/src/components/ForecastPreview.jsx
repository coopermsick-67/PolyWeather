import { ArrowRight, CloudSun, Sun } from 'lucide-react'

const PREVIEW_LIMIT = 6

export default function ForecastPreview({ data, loading, error }) {
  const forecasts = data?.forecasts ?? []
  const shown = forecasts.slice(0, PREVIEW_LIMIT)
  const remaining = forecasts.length - shown.length
  return <section className="preview-card" aria-label="Daily high forecast preview"><div className="preview-card-top"><div><span>EXPERIMENTAL FORECAST</span><strong>{data?.targetDate ? new Date(`${data.targetDate}T12:00:00`).toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' }) : 'Loading today'}</strong></div><Sun aria-hidden="true" /></div>{loading && !forecasts.length ? <div className="preview-loading"><i /><i /><i /><i /><i /></div> : forecasts.length ? <div className="preview-rows">{shown.map((forecast) => <div key={forecast.station}><div><span>{forecast.station}</span><strong>{forecast.city.split(',')[0]}</strong></div><Sun aria-hidden="true" /><b>{forecast.highF}°</b><small>{forecast.fourDegreeRangeLowF}–{forecast.fourDegreeRangeHighF}°</small></div>)}</div> : <div className="preview-error"><CloudSun /><strong>Live forecast unavailable</strong><span>{error || 'Open the dashboard to refresh.'}</span></div>}<a href="#/forecast">{remaining > 0 ? `See all ${forecasts.length} locations` : 'Open forecast workspace'} <ArrowRight size={15} /></a></section>
}
