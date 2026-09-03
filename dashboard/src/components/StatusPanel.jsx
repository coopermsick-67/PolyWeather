import { CheckCircle2, Clock3, Database, MapPin } from 'lucide-react'

export default function StatusPanel({ data, loading }) {
  const generated = data?.generatedAt ? new Date(data.generatedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'Waiting for live data'
  const stationCount = data?.stationRegistry?.length
  return <section className="side-card status-card" aria-labelledby="status-title"><div className="side-title"><CheckCircle2 /><h2 id="status-title">Forecast status</h2></div><div className={loading ? 'status-line muted' : 'status-line'}><CheckCircle2 size={17} /><div><strong>{loading ? 'Updating forecast' : data?.modelStatus ?? 'Experimental forecast guidance'}</strong><span>{loading ? 'Reading the latest available guidance.' : data?.forecastInputs ?? 'Waiting for forecast-input metadata.'}</span></div></div><dl><div><dt><Clock3 size={15} />Last refresh</dt><dd>{generated}<small>Your device time</small></dd></div><div><dt><MapPin size={15} />Locations</dt><dd>{stationCount ? `${stationCount} weather stations` : 'Loading…'}</dd></div><div><dt><Database size={15} />Validation target</dt><dd>{data?.validationTarget ?? '—'}</dd></div></dl><p className="side-foot">{data?.releaseStatus ?? data?.evaluationContract ?? 'Model status will appear after the first refresh.'}</p></section>
}
