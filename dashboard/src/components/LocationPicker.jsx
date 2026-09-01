import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, MapPin, Search } from 'lucide-react'

export default function LocationPicker({ stations, value, onChange, allowAll = false, allLabel = 'All stations', calibratedIds, className = '' }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDocClick = (event) => { if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false) }
    const onKey = (event) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  useEffect(() => {
    if (open) {
      setQuery('')
      window.setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return stations
    return stations.filter((station) => `${station.display_name} ${station.stationId} ${station.name}`.toLowerCase().includes(normalized))
  }, [stations, query])

  const current = stations.find((station) => station.stationId === value)
  const label = value == null ? allLabel : current ? `${current.display_name} · ${current.stationId}` : value

  const select = (next) => {
    onChange(next)
    setOpen(false)
  }

  return <div className={`location-picker ${className}`} ref={rootRef}>
    <button type="button" className="location-picker-trigger" onClick={() => setOpen((prev) => !prev)} aria-haspopup="listbox" aria-expanded={open}>
      <MapPin size={15} aria-hidden="true" />
      <span>{label}</span>
      <ChevronDown size={15} aria-hidden="true" className={open ? 'flip' : ''} />
    </button>
    {open && <div className="location-picker-panel" role="listbox" aria-label="Choose a settlement station">
      <div className="location-picker-search"><Search size={14} aria-hidden="true" /><input ref={inputRef} type="text" placeholder="Search city or station code" value={query} onChange={(event) => setQuery(event.target.value)} /></div>
      <div className="location-picker-list">
        {allowAll && <button type="button" role="option" aria-selected={value == null} className={value == null ? 'selected' : ''} onClick={() => select(null)}><span className="location-picker-name">{allLabel}</span></button>}
        {filtered.length === 0 && <p className="location-picker-empty">No locations match “{query}”.</p>}
        {filtered.map((station) => {
          const calibrated = calibratedIds?.has(station.stationId)
          return <button key={station.stationId} type="button" role="option" aria-selected={value === station.stationId} className={value === station.stationId ? 'selected' : ''} onClick={() => select(station.stationId)}>
            <span className={calibratedIds ? `location-picker-dot ${calibrated ? 'is-calibrated' : ''}` : 'sr-only'} aria-hidden="true" title={calibratedIds ? (calibrated ? 'Calibrated model' : 'Live baseline, not yet calibrated') : undefined} />
            <span className="location-picker-name">{station.display_name}<small>{station.name}</small></span>
            <span className="location-picker-code">{station.stationId}</span>
          </button>
        })}
      </div>
    </div>}
  </div>
}
