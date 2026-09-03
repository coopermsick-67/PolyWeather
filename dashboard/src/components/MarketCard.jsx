import { AlertTriangle, BadgeInfo, Radio, ShieldAlert } from 'lucide-react'
import LocationPicker from './LocationPicker'
import { stationTime } from '../dateUtils'

function statusClass(status) {
  return status === 'PROVISIONAL' ? 'status-provisional' : 'status-caution'
}

export default function MarketCard({ registry = [], selectedStation, forecast, onSelectStation, calibratedIds }) {
  const station = registry.find((item) => item.stationId === selectedStation) ?? registry[0]
  if (!station) return null
  const hasForecast = Boolean(forecast)
  const hasSpread = Number.isFinite(forecast?.modelSpreadF)
  return <section className="market-card" aria-labelledby="market-card-title">
    <div className="market-card-head">
      <div><p className="section-label">SETTLEMENT-STATION MARKET</p><h2 id="market-card-title">{station.display_name} <span>/ {station.stationId}</span></h2></div>
      <div className="market-location"><span className="market-location-label">Location</span><LocationPicker stations={registry} value={station.stationId} onChange={onSelectStation} calibratedIds={calibratedIds} /></div>
    </div>
    <div className="settlement-warning"><AlertTriangle size={16} /><span><strong>Settlement station:</strong> {station.display_note}. Never substitute a city-center or similarly named airport.</span></div>
    <div className="market-metrics">
      <div className="market-estimate"><span>Daily high estimate</span><strong>{hasForecast ? `${forecast.highF}°F` : '—'}</strong><small>{hasForecast ? `${forecast.rangeLowF}°–${forecast.rangeHighF}° displayed uncertainty band` : 'No calibrated model is available yet.'}</small></div>
      <div><span>Current observation</span><strong>{forecast?.currentObservedTemperatureF ?? '—'}{forecast?.currentObservedTemperatureF != null ? '°F' : ''}</strong><small>{forecast?.lastObservationAt ? `Updated ${stationTime(forecast.lastObservationAt, station.timezone)} ${station.timezone}` : 'No live observation loaded'}</small></div>
      <div><span>Live model spread</span><strong>{hasSpread ? `${forecast.modelSpreadF.toFixed(1)}°F` : '—'}</strong><small>{hasSpread ? 'Difference across available model highs; not a probability.' : forecast?.sourceName ?? 'Awaiting source details.'}</small></div>
      <div><span>Data-quality status</span><strong className={`market-status ${statusClass(forecast?.dataQualityStatus)}`}>{forecast?.dataQualityStatus ?? 'NO BET / INSUFFICIENT DATA'}</strong><small>{hasForecast ? (forecast.isCalibrated ? 'Historical score exists; no market contract or executable price is verified.' : 'Guidance shown without historical residual validation.') : 'Configured station; not trained or rule-linked.'}</small></div>
    </div>
    {hasForecast && <div className="market-explain"><BadgeInfo size={16} /><span>{forecast.reasonCodes?.[0] || 'Current prediction is still experimental and must not be treated as settlement.'}</span></div>}
    {!hasForecast && <div className="market-explain"><ShieldAlert size={16} /><span>This station is mapped correctly but has no validated historical calibration artifact. The app will not manufacture a strong prediction.</span></div>}
    <div className="market-foot"><Radio size={14} /><span>Timezone: {station.timezone} · NWS grid: {station.forecastGrid.office}/{station.forecastGrid.x},{station.forecastGrid.y}</span></div>
  </section>
}
