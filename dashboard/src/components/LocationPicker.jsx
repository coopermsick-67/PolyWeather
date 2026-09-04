import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, MapPin, Search } from 'lucide-react'

export default function LocationPicker({ stations, value, onChange, allowAll = false, allLabel = 'All stations', calibratedIds, className = '' }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(-1)
  const rootRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDocClick = (event) => { if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  useEffect(() => {
    if (open) {
      setQuery('')
      setActiveIndex(-1)
      window.setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return stations
    return stations.filter((station) => `${station.display_name} ${station.stationId} ${station.name}`.toLowerCase().includes(normalized))
  }, [stations, query])

  // The typed query can shrink the list on every keystroke; clamp instead of
  // leaving activeIndex pointing at an option that no longer exists.
  useEffect(() => {
    setActiveIndex((prev) => {
      const optionCount = filtered.length + (allowAll ? 1 : 0)
      return prev >= optionCount ? optionCount - 1 : prev
    })
  }, [filtered, allowAll])

  const options = useMemo(
    () => (allowAll ? [{ stationId: null, display_name: allLabel, name: '' }, ...filtered] : filtered),
    [allowAll, allLabel, filtered]
  )

  const current = stations.find((station) => station.stationId === value)
  const label = value == null ? allLabel : current ? `${current.display_name} · ${current.stationId}` : value

  const select = (next) => {
    onChange(next)
    setOpen(false)
  }

  const onKeyDown = (event) => {
    if (event.key === 'Escape') { setOpen(false); return }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex((prev) => (options.length ? (prev + 1) % options.length : -1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex((prev) => (options.length ? (prev - 1 + options.length) % options.length : -1))
    } else if (event.key === 'Enter') {
      if (activeIndex >= 0 && activeIndex < options.length) {
        event.preventDefault()
        select(options[activeIndex].stationId)
      }
    }
  }

  return <div className={`location-picker ${className}`} ref={rootRef}>
    <button type="button" className="location-picker-trigger" onClick={() => setOpen((prev) => !prev)} aria-haspopup="listbox" aria-expanded={open}>
      <MapPin size={15} aria-hidden="true" />
      <span>{label}</span>
      <ChevronDown size={15} aria-hidden="true" className={open ? 'flip' : ''} />
    </button>
    {open && <div className="location-picker-panel" role="listbox" aria-label="Choose a settlement station" id="location-picker-listbox">
      <div className="location-picker-search"><Search size={14} aria-hidden="true" /><input ref={inputRef} type="text" placeholder="Search city or station code" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={onKeyDown} role="combobox" aria-expanded="true" aria-controls="location-picker-listbox" aria-activedescendant={activeIndex >= 0 ? `location-picker-option-${activeIndex}` : undefined} /></div>
      <div className="location-picker-list">
        {allowAll && <button id="location-picker-option-0" type="button" role="option" aria-selected={value == null} className={[value == null && 'selected', activeIndex === 0 && 'is-active'].filter(Boolean).join(' ')} onClick={() => select(null)}><span className="location-picker-name">{allLabel}</span></button>}
        {filtered.length === 0 && <p className="location-picker-empty">{stations.length === 0 ? 'Loading locations…' : `No locations match "${query}".`}</p>}
        {filtered.map((station, index) => {
          const calibrated = calibratedIds?.has(station.stationId)
          const optionIndex = allowAll ? index + 1 : index
          return <button id={`location-picker-option-${optionIndex}`} key={station.stationId} type="button" role="option" aria-selected={value === station.stationId} className={[value === station.stationId && 'selected', activeIndex === optionIndex && 'is-active'].filter(Boolean).join(' ')} onClick={() => select(station.stationId)}>
            <span className={calibratedIds ? `location-picker-dot ${calibrated ? 'is-calibrated' : ''}` : 'sr-only'} aria-hidden="true" title={calibratedIds ? (calibrated ? 'Calibrated model' : 'Live baseline, not yet calibrated') : undefined} />
            <span className="location-picker-name">{station.display_name}<small>{station.name}</small></span>
            <span className="location-picker-code">{station.stationId}</span>
          </button>
        })}
      </div>
    </div>}
  </div>
}
