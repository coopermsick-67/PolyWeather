// Pure helpers shared between the test suite and the generated Cloudflare
// Worker (dashboard/scripts/create-worker.mjs). The build script reads this
// file's source text and splices it verbatim into the generated worker
// bundle -- these are not just documentation copies that could drift, they
// are the exact code that ships. Keep every export free of imports and
// external state so a straight text-splice into the Worker template stays
// valid.

export function localDate(value, timeZone) {
  const parts = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: '2-digit', day: '2-digit', timeZone }).formatToParts(value)
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))
  return values.year + '-' + values.month + '-' + values.day
}

export function addDays(iso, days) {
  const date = new Date(iso + 'T12:00:00Z')
  date.setUTCDate(date.getUTCDate() + days)
  return date.toISOString().slice(0, 10)
}

export function isIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value ?? '')) return false
  const [year, month, day] = value.split('-').map(Number)
  const date = new Date(Date.UTC(year, month - 1, day))
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day
}

// The lower bound is the earliest "today" across every configured station's
// own timezone, not just America/New_York's. A West Coast station's local
// calendar day has not yet rolled over for up to ~3 hours after
// America/New_York already has (9-11:59 PM Pacific is already past
// midnight Eastern); anchoring the lower bound to Eastern alone would
// reject a still-current Pacific "today" during that window.
export function targetDateForRequest(value, today, earliestToday = today) {
  if (!value) return today
  const maxDate = addDays(today, 7)
  if (!isIsoDate(value) || value < earliestToday || value > maxDate) {
    const error = new Error('Forecast date must be between ' + earliestToday + ' and ' + maxDate + '.')
    error.status = 400
    throw error
  }
  return value
}

export function earliestStationLocalToday(now, stations) {
  let earliest = null
  for (const station of stations) {
    const local = localDate(now, station.timezone)
    if (earliest === null || local < earliest) earliest = local
  }
  return earliest
}

export function hex(value) {
  return Array.from(new Uint8Array(value)).map((byte) => byte.toString(16).padStart(2, '0')).join('')
}

export function constantTimeEqual(a, b) {
  if (a.length !== b.length) return false
  let mismatch = 0
  for (let index = 0; index < a.length; index++) mismatch |= a.charCodeAt(index) ^ b.charCodeAt(index)
  return mismatch === 0
}
