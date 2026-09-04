import { Clipboard, SlidersHorizontal } from 'lucide-react'
import LocationPicker from './LocationPicker'

export default function ForecastControls({ registry, selectedStation, onSelectStation, calibratedIds, showBaseline, onShowBaseline, onCopy }) {
  return <section className="forecast-controls" aria-label="Forecast controls">
    <div className="station-filter"><SlidersHorizontal size={16} aria-hidden="true" /><span>Station</span><LocationPicker stations={registry} value={selectedStation} onChange={onSelectStation} allowAll allLabel="All stations" calibratedIds={calibratedIds} className="station-filter-picker" /><small>{registry.length || '—'} mapped</small></div>
    <div className="control-actions"><label className="baseline-switch"><input type="checkbox" checked={showBaseline} onChange={(event) => onShowBaseline(event.target.checked)} /><span aria-hidden="true" /><b>Show NBM baseline</b></label><button className="copy-button" type="button" onClick={onCopy}><Clipboard size={15} />Copy planning range</button></div>
  </section>
}
