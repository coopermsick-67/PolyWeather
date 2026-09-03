const dimensions = { width: 1120, height: 235, left: 28, right: 18, top: 22, bottom: 30 }

function points(values, max = 3) {
  const { width, height, left, right, top, bottom } = dimensions
  const usableWidth = width - left - right
  const usableHeight = height - top - bottom
  return values.map((value, index) => {
    const x = left + (index / (values.length - 1)) * usableWidth
    const y = top + (1 - value / max) * usableHeight
    return `${x},${y}`
  }).join(' ')
}

export default function TrendChart({ trend }) {
  if (!trend?.candidate?.length || !trend?.baseline?.length || !trend?.labels?.length) {
    return <div className="chart-empty" role="status">Performance history will appear after the forecast data loads.</div>
  }
  const { width, height, left, right, top, bottom } = dimensions
  const grid = [0, 1, 2, 3]
  const candidate = points(trend.candidate)
  const baseline = points(trend.baseline)
  const labelEvery = Math.max(1, Math.floor(trend.labels.length / 7))

  return (
    <section className="trend-panel" aria-labelledby="trend-heading">
      <div className="trend-heading">
        <div><h2 id="trend-heading">Shadow monitoring</h2><p>14-day rolling error · mean absolute error (°F)</p></div>
        <div className="legend"><span><i className="candidate" />WeatherPicks candidate</span><span><i className="baseline" />Raw NBM baseline</span></div>
      </div>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Rolling forecast error trend">
          {grid.map((tick) => {
            const y = top + (1 - tick / 3) * (height - top - bottom)
            return <g key={tick}><line x1={left} x2={width - right} y1={y} y2={y} className="grid" /><text x="0" y={y + 4}>{tick.toFixed(1)}</text></g>
          })}
          <polyline points={baseline} className="baseline-line" />
          <polyline points={candidate} className="candidate-line" />
          {trend.candidate.map((value, index) => {
            const actualX = left + (index / (trend.candidate.length - 1)) * (width - left - right)
            const actualY = top + (1 - value / 3) * (height - top - bottom)
            return <circle key={index} cx={actualX} cy={actualY} r="3.6" className="candidate-dot" />
          })}
          {trend.labels.map((label, index) => {
            if (index % labelEvery !== 0 && index !== trend.labels.length - 1) return null
            const x = left + (index / (trend.labels.length - 1)) * (width - left - right)
            return <text key={label} x={x} y={height - 6} textAnchor="middle" className="axis-label">{label}</text>
          })}
        </svg>
      </div>
    </section>
  )
}
