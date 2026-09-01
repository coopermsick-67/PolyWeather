import { Info } from 'lucide-react'

export default function ForecastBrief({ evidence }) {
  const fourDegreeCoverage = evidence?.fourDegreeCoveragePct ?? '—'
  const calibratedCoverage = evidence?.calibratedCoveragePct ?? '—'
  const calibratedWidth = evidence?.calibratedMeanWidthF ?? '—'
  return <section className="side-card forecast-brief" aria-labelledby="confidence-title"><div className="side-title"><Info /><h2 id="confidence-title">Range, honestly</h2></div><p><strong>The default planning range is ±2°F</strong>—a four-degree span around the displayed high. It contained the observed high in {fourDegreeCoverage}% of the untouched historical candidate backtest.</p><div className="brief-key"><span><b /><b /><b /><b /></span><div><strong>Best 4° planning range</strong><p>Useful for everyday planning; it is not a confidence guarantee.</p></div></div><div className="brief-key"><span><b /><b /><b className="muted" /><b className="muted" /></span><div><strong>Calibrated model range</strong><p>The station-specific {calibratedCoverage}% nominal band averages {calibratedWidth}° wide and is shown in station details.</p></div></div><p className="side-foot">Experimental shadow model. If a higher temperature has already been observed today, the display cannot be lower than that observation.</p></section>
}
