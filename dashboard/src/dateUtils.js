export function formatIso(date) {
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day}`
}

export function isoToday() {
  const now = new Date()
  return formatIso(new Date(now.getFullYear(), now.getMonth(), now.getDate()))
}

export function addDays(iso, days) {
  const date = new Date(`${iso}T12:00:00`)
  date.setDate(date.getDate() + days)
  return formatIso(date)
}

export function longDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })
}

export function shortDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function dateChoice(iso, today) {
  const offset = Math.round((new Date(`${iso}T12:00:00`) - new Date(`${today}T12:00:00`)) / 86400000)
  if (offset === 0) return 'Today'
  if (offset === 1) return 'Tomorrow'
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' })
}

function formatParts(value, timeZone) {
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(new Date(value)).reduce((parts, part) => {
    if (part.type !== 'literal') parts[part.type] = Number(part.value)
    return parts
  }, {})
}

// Convert a station-local calendar date to its first instant. This avoids
// borrowing the viewer's timezone for the intraday chart, including DST days.
export function stationDayStart(iso, timeZone) {
  const [year, month, day] = iso.split('-').map(Number)
  const guess = Date.UTC(year, month - 1, day)
  const local = formatParts(guess, timeZone)
  const localAsUtc = Date.UTC(local.year, local.month - 1, local.day, local.hour, local.minute, local.second)
  return guess - (localAsUtc - guess)
}

export function stationDayBounds(iso, timeZone) {
  return { start: stationDayStart(iso, timeZone), end: stationDayStart(addDays(iso, 1), timeZone) }
}

export function stationIsoDate(value, timeZone) {
  const parts = formatParts(value, timeZone)
  return `${parts.year}-${String(parts.month).padStart(2, '0')}-${String(parts.day).padStart(2, '0')}`
}

export function stationTime(value, timeZone) {
  return new Intl.DateTimeFormat('en-US', { timeZone, hour: 'numeric', minute: '2-digit' }).format(new Date(value))
}
