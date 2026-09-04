import { useEffect, useRef, useState } from 'react'
import { stationDayBounds, stationIsoDate, stationTime } from '../dateUtils'

const CHART_LEFT = 46
const CHART_WIDTH = 810
const SVG_WIDTH = 880

function path(values, x, y) { return values.map((point, index) => `${index ? 'L' : 'M'}${x(point.x).toFixed(1)},${y(point.y).toFixed(1)}`).join(' ') }

function nearestPoint(points, targetX) {
  if (!points.length) return null
  return points.reduce((best, point) => Math.abs(point.x - targetX) < Math.abs(best.x - targetX) ? point : best, points[0])
}

export default function IntradayChart({ forecast }) {
  const [hoverX, setHoverX] = useState(null)
  const scrollRef = useRef(null)
  const rafRef = useRef(null)

  useEffect(() => { setHoverX(null) }, [forecast?.station])
  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }, [])

  useEffect(() => {
    const container = scrollRef.current
    if (!container || !forecast) return
    const timezone = forecast.timezone || 'UTC'
    const { start: dayStart, end: dayEnd } = stationDayBounds(forecast.targetDate || stationIsoDate(Date.now(), timezone), timezone)
    const now = Date.now()
    if (now < dayStart || now >= dayEnd || container.scrollWidth <= container.clientWidth) {
      container.scrollLeft = 0
      return
    }
    const nowX = CHART_LEFT + ((now - dayStart) / (dayEnd - dayStart)) * CHART_WIDTH
    const scale = container.scrollWidth / SVG_WIDTH
    container.scrollLeft = Math.max(0, nowX * scale - container.clientWidth / 2)
  }, [forecast?.station, forecast?.targetDate])

  if (!forecast) return null
  const timezone = forecast.timezone || 'UTC'
  const observations = (forecast.intradayObservations || []).map((item) => ({ x: new Date(item.time).getTime(), y: item.temperatureF })).filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y)).sort((left, right) => left.x - right.x)
  const now = Date.now()
  const targetDate = forecast.targetDate || stationIsoDate(now, timezone)
  const { start: dayStart, end: dayEnd } = stationDayBounds(targetDate, timezone)
  const dayDuration = dayEnd - dayStart
  const isToday = targetDate === stationIsoDate(now, timezone)
  const low = Math.floor(Math.min(forecast.rangeLowF, forecast.baselineHighF, ...(observations.map((item) => item.y)), forecast.currentObservedTemperatureF ?? forecast.rangeLowF) - 3)
  const high = Math.ceil(Math.max(forecast.rangeHighF, forecast.baselineHighF, ...(observations.map((item) => item.y)), forecast.currentObservedTemperatureF ?? forecast.rangeHighF) + 3)
  const chartBottom = 170
  const chartTop = 38
  const x = (value) => CHART_LEFT + ((Math.max(dayStart, Math.min(dayEnd, value)) - dayStart) / dayDuration) * CHART_WIDTH
  const y = (value) => chartBottom - ((value - low) / Math.max(1, high - low)) * (chartBottom - chartTop)
  const last = observations.at(-1)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => dayStart + fraction * dayDuration)
  const hoverTime = hoverX == null ? null : dayStart + (hoverX / CHART_WIDTH) * dayDuration
  const hoverPoint = hoverTime == null ? null : nearestPoint(observations, hoverTime)
  const onPointerMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const clientX = event.clientX
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      const svgX = ((clientX - rect.left) / rect.width) * SVG_WIDTH
      setHoverX(Math.max(0, Math.min(CHART_WIDTH, svgX - CHART_LEFT)))
    })
  }
  const sourceStatus = forecast.dataFreshness >= 1 ? 'Source response received' : 'Source response stale'
  const hasSpread = Number.isFinite(forecast.modelSpreadF)
  return <section className="intraday-card" aria-labelledby="intraday-title">
    <div className="intraday-heading">
      <div><p className="section-label">LIVE STATION TRACE</p><h2 id="intraday-title">Intraday temperature</h2><span>{forecast.marketLocation} · {forecast.station}</span></div>
      <div className="intraday-evidence">
        <div className="evidence-metric"><span>Live model spread</span><strong>{hasSpread ? `${forecast.modelSpreadF.toFixed(1)}°F` : '—'}</strong></div>
        <div className={`evidence-pill ${forecast.dataFreshness >= 1 ? 'is-fresh' : 'is-stale'}`}>{sourceStatus}</div>
      </div>
    </div>
    <div className="intraday-legend"><span><i className="observed" />Observed</span><span><i className="projected" />Daily-high estimate</span><span><i className="baseline" />NBM baseline</span><span><i className="band" />Displayed uncertainty band</span></div>
    <div className="intraday-chart" ref={scrollRef}>
      <svg viewBox="0 0 880 200" role="img" aria-label={`Live intraday temperature chart for ${forecast.station}`} onPointerMove={onPointerMove} onPointerLeave={() => setHoverX(null)}>
        {[low, Math.round((low + high) / 2), high].map((tick) => <g key={tick}><line x1={CHART_LEFT} x2="856" y1={y(tick)} y2={y(tick)} className="intraday-grid" /><text x="9" y={y(tick) + 4}>{tick}°</text></g>)}
        <rect x={CHART_LEFT} y={y(forecast.rangeHighF)} width={CHART_WIDTH} height={Math.max(1, y(forecast.rangeLowF) - y(forecast.rangeHighF))} className="range-band" />
        <line x1={CHART_LEFT} x2="856" y1={y(forecast.baselineHighF)} y2={y(forecast.baselineHighF)} className="baseline-line" />
        <line x1={CHART_LEFT} x2="856" y1={y(forecast.highF)} y2={y(forecast.highF)} className="estimate-line" />
        {observations.length > 1 && <path d={path(observations, x, y)} className="observed-line" />}
        {last && <circle cx={x(last.x)} cy={y(last.y)} r="4" className="current-dot" />}
        {isToday && <g className="now-marker"><line x1={x(now)} x2={x(now)} y1={chartTop} y2={chartBottom} /><text x={x(now)} y={chartTop - 6} textAnchor="middle">Now</text></g>}
        {hoverPoint && <g className="hover-marker"><line x1={x(hoverPoint.x)} x2={x(hoverPoint.x)} y1={chartTop} y2={chartBottom} /><circle cx={x(hoverPoint.x)} cy={y(hoverPoint.y)} r="3.5" /></g>}
        {ticks.map((time) => <text key={time} x={x(time)} y="193" textAnchor="middle" className="intraday-axis">{stationTime(time, timezone)}</text>)}
      </svg>
      {hoverPoint && <div className="intraday-tooltip" style={{ left: `${(x(hoverPoint.x) / SVG_WIDTH) * 100}%` }}><strong>{Math.round(hoverPoint.y)}°F</strong><span>{stationTime(hoverPoint.x, timezone)} {timezone}</span></div>}
    </div>
    {observations.length < 2 && <p className="intraday-data-note">Limited intraday readings are available for this station. The estimate and displayed range remain visible, but the observed-temperature trace will fill in after the next station update.</p>}
    <div className="intraday-stats">
      <span>Current <strong>{forecast.currentObservedTemperatureF ?? '—'}{forecast.currentObservedTemperatureF != null ? '°F' : ''}</strong></span>
      <span>High so far <strong>{forecast.observedHighSoFarF ?? '—'}{forecast.observedHighSoFarF != null ? '°F' : ''}</strong></span>
      <span>Low so far <strong>{forecast.observedLowSoFarF ?? '—'}{forecast.observedLowSoFarF != null ? '°F' : ''}</strong></span>
      {forecast.isCalibrated ? <>
        <span>Live NBM baseline <strong>{forecast.baselineHighF}°F</strong></span>
        <span>Calibrated high <strong>{forecast.highF}°F</strong></span>
      </> : <span>Live NBM baseline <strong>{forecast.baselineHighF}°F</strong><small>Shown as-is; not yet calibrated</small></span>}
      <span>4° planning range <strong>{forecast.fourDegreeRangeLowF}°–{forecast.fourDegreeRangeHighF}°</strong><small>Not a live-calibrated interval</small></span>
    </div>
    {forecast.reasonCodes?.length > 0 && <ul className="intraday-reasons">{forecast.reasonCodes.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
  </section>
}
