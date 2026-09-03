import { Info } from 'lucide-react'

export default function ForecastBrief({ evidence }) {
  const fourDegreeCoverage = evidence?.fourDegreeCoveragePct ?? '—'
  const calibratedCoverage = evidence?.calibratedCoveragePct ?? '—'
  const calibratedWidth = evidence?.calibratedMeanWidthF ?? '—'
  return <section className="side-card forecast-brief" aria-labelledby="confidence-title"><div className="side-title"><Info /><h2 id="confidence-title">Range, honestly</h2></div><p><strong>The default planning range is ±2°F</strong>—a four-degree span around the displayed high. It contained the observed high in {fourDegreeCoverage}% of the archived-composite historical evaluation, not as a guarantee for a live refresh.</p><div className="brief-key"><span><b /><b /><b /><b /></span><div><strong>4° planning range</strong><p>Useful for everyday planning; it is not a confidence guarantee.</p></div></div><div className="brief-key"><span><b /><b /><b className="muted" /><b className="muted" /></span><div><strong>Historical P10–P90 band</strong><p>The station-specific 80% nominal archive band covered {calibratedCoverage}% of outcomes and averaged {calibratedWidth}° wide. The live display band has not been prospectively calibrated.</p></div></div><p className="side-foot">Experimental shadow model. If a higher temperature has already been observed today, the display cannot be lower than that observation.</p></section>
}
