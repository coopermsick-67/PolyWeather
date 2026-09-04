import { stationDayBounds, stationIsoDate, stationTime } from '../dateUtils'

const LEFT = 46
const WIDTH = 720
const RIGHT = LEFT + WIDTH
const TOP = 31
const BOTTOM = 194

function linePath(points, x, y) {
  return points.map((point, index) => `${index ? 'L' : 'M'}${x(point.time).toFixed(1)},${y(point.temperature).toFixed(1)}`).join(' ')
}

const GAUGE_RADIUS = 42
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS

function AgreementGauge({ value }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value))
  const dash = (pct / 100) * GAUGE_CIRCUMFERENCE
  const tone = value == null ? 'unknown' : value >= 80 ? 'high' : value >= 55 ? 'mid' : 'low'
  return <div className={`agreement-gauge tone-${tone}`}>
    <svg viewBox="0 0 100 100" role="img" aria-label={value == null ? 'Source agreement unavailable' : `Source agreement ${value}%`}>
      <circle cx="50" cy="50" r={GAUGE_RADIUS} className="agreement-gauge-track" />
      <circle cx="50" cy="50" r={GAUGE_RADIUS} className="agreement-gauge-value" strokeDasharray={`${dash} ${GAUGE_CIRCUMFERENCE}`} transform="rotate(-90 50 50)" />
    </svg>
    <div className="agreement-gauge-label"><strong>{value == null ? '—' : `${value}%`}</strong><span>{tone === 'high' ? 'High' : tone === 'mid' ? 'Moderate' : tone === 'low' ? 'Low' : 'Unknown'}</span></div>
  </div>
}

function usableObservations(forecast) {
  const points = (forecast.intradayObservations || [])
    .map((point) => ({ time: new Date(point.time).getTime(), temperature: Number(point.temperatureF) }))
    .filter((point) => Number.isFinite(point.time) && Number.isFinite(point.temperature))
    .sort((left, right) => left.time - right.time)

  // Preserve the first and last readings while keeping the SVG responsive on busy stations.
  if (points.length <= 160) return points
  const stride = Math.ceil(points.length / 160)
  return points.filter((_, index) => index === 0 || index === points.length - 1 || index % stride === 0)
}

export default function ForecastFanChart({ forecast }) {
  if (!forecast) return null

  const timezone = forecast.timezone || 'UTC'
  const now = Date.now()
  const targetDate = forecast.targetDate || stationIsoDate(now, timezone)
  const { start, end } = stationDayBounds(targetDate, timezone)
  const observations = usableObservations(forecast)
  const observedValues = observations.map((point) => point.temperature)
  const candidates = [
    forecast.rangeLowF, forecast.rangeHighF, forecast.fourDegreeRangeLowF, forecast.fourDegreeRangeHighF,
    forecast.highF, forecast.baselineHighF, forecast.observedHighSoFarF, ...observedValues,
  ].filter(Number.isFinite)
  const low = Math.floor(Math.min(...candidates) - 3)
  const high = Math.ceil(Math.max(...candidates) + 3)
  const duration = Math.max(1, end - start)
  const x = (time) => LEFT + ((Math.max(start, Math.min(end, time)) - start) / duration) * WIDTH
  const y = (temperature) => BOTTOM - ((temperature - low) / Math.max(1, high - low)) * (BOTTOM - TOP)
  const isToday = targetDate === stationIsoDate(now, timezone)
  const ticks = [0, .25, .5, .75, 1].map((part) => start + part * duration)
  const agreement = Number.isFinite(forecast.sourceAgreement) ? Math.round(Math.max(0, Math.min(1, forecast.sourceAgreement)) * 100) : null
  const spread = Number.isFinite(forecast.modelSpreadF) ? forecast.modelSpreadF.toFixed(1) : null
  const highSoFar = Number.isFinite(forecast.observedHighSoFarF) ? forecast.observedHighSoFarF : null
  const hasObservationTrace = observations.length > 1
  const hasBand = Number.isFinite(forecast.rangeLowF) && Number.isFinite(forecast.rangeHighF)

  return <section className="forecast-fan-card" aria-labelledby="forecast-fan-title">
    <div className="forecast-fan-heading">
      <div>
        <p className="section-label">DAILY HIGH GUIDE</p>
        <h2 id="forecast-fan-title">{forecast.city} forecast trace</h2>
        <p>Observed station temperatures against the selected day’s daily-high guidance.</p>
      </div>
      <div className="forecast-fan-summary">
        <span>Daily estimate <strong>{forecast.highF}°F</strong></span>
        <span>NBM baseline <strong>{forecast.baselineHighF}°F</strong></span>
      </div>
    </div>

    <div className="forecast-fan-layout">
      <div className="forecast-fan-plot">
        <div className="forecast-fan-legend" aria-label="Forecast graph legend">
          <span><i className="fan-observed" />Observed temperature</span>
          <span><i className="fan-estimate" />Daily estimate</span>
          <span><i className="fan-baseline" />NBM baseline</span>
          <span><i className="fan-band" />{hasBand ? 'Calibrated interval' : 'Interval unavailable'}</span>
        </div>
        <div className="forecast-fan-svg-wrap">
          <svg viewBox="0 0 810 222" role="img" aria-label={`Daily high forecast graph for ${forecast.station}. Estimate ${forecast.highF} degrees Fahrenheit, NBM baseline ${forecast.baselineHighF} degrees Fahrenheit.`}>
            {[low, Math.round((low + high) / 2), high].map((tick) => <g key={tick}><line className="fan-grid" x1={LEFT} x2={RIGHT} y1={y(tick)} y2={y(tick)} /><text x="8" y={y(tick) + 4}>{tick}°</text></g>)}
            {hasBand && <rect className="fan-band-area" x={LEFT} y={y(forecast.rangeHighF)} width={WIDTH} height={Math.max(1, y(forecast.rangeLowF) - y(forecast.rangeHighF))} />}
            <line className="fan-baseline-line" x1={LEFT} x2={RIGHT} y1={y(forecast.baselineHighF)} y2={y(forecast.baselineHighF)} />
            <line className="fan-estimate-line" x1={LEFT} x2={RIGHT} y1={y(forecast.highF)} y2={y(forecast.highF)} />
            {hasObservationTrace && <path className="fan-observed-line" d={linePath(observations, x, y)} />}
            {highSoFar != null && <line className="fan-high-so-far" x1={LEFT} x2={isToday ? x(now) : RIGHT} y1={y(highSoFar)} y2={y(highSoFar)} />}
            {isToday && now > start && now < end && <g className="fan-now"><line x1={x(now)} x2={x(now)} y1={TOP} y2={BOTTOM} /><text x={x(now)} y={TOP - 8} textAnchor="middle">Now</text></g>}
            {ticks.map((time) => <text className="fan-axis" key={time} x={x(time)} y="215" textAnchor="middle">{stationTime(time, timezone)}</text>)}
          </svg>
        </div>
        <p className="forecast-fan-note">{hasBand ? 'The blue band is the calibrated daily-high interval, not a projected hour-by-hour path.' : 'No calibrated uncertainty interval is available for this fallback value.'} {hasObservationTrace ? 'The white line is reported station temperature.' : 'No intraday trace is available for this selected date yet.'}</p>
      </div>

      <aside className="forecast-fan-evidence" aria-label="Source agreement">
        <span className="forecast-fan-evidence-label">SOURCE AGREEMENT</span>
        <AgreementGauge value={agreement} />
        <p>{spread == null ? 'Model spread has not been reported.' : `${spread}°F spread across available source highs.`}</p>
        <dl>
          <div><dt>Calibrated interval</dt><dd>{hasBand ? `${forecast.rangeLowF}°–${forecast.rangeHighF}°` : 'Unavailable'}</dd></div>
          <div><dt>Observed high so far</dt><dd>{highSoFar == null ? '—' : `${highSoFar}°F`}</dd></div>
        </dl>
        <small>Agreement measures the spread of available guidance, not the chance of a correct settlement.</small>
      </aside>
    </div>
  </section>
}
