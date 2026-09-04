const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export async function getBacktestArchive() {
  const response = await fetch(`${API_BASE_URL}/backtest-data.json`, { cache: 'no-store' })
  if (!response.ok) throw new Error('Backtest archive is unavailable.')
  return response.json()
}

async function requestDashboard(date, force, signal) {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 30_000)
  const abortFromCaller = () => controller.abort()
  signal?.addEventListener('abort', abortFromCaller, { once: true })
  try {
    const url = `${API_BASE_URL}/api/dashboard?date=${encodeURIComponent(date)}${force ? '&refresh=1' : ''}`
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' })
    const contentType = response.headers.get('content-type') || ''
    const payload = contentType.includes('application/json') ? await response.json() : null
    if (!response.ok) {
      const error = new Error(payload?.error || `Forecast service returned ${response.status}.`)
      error.status = response.status
      throw error
    }
    if (!payload) throw new Error('Forecast service returned an invalid response.')
    return payload
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) throw new Error('Forecast request timed out. Try refreshing in a moment.')
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abortFromCaller)
  }
}

function isTransientFailure(error) {
  // A 4xx (bad date, validation) will not succeed on retry; a network
  // failure or 5xx from an upstream hiccup (NWS timeout, cold start) often
  // will, so only those are worth one automatic retry before surfacing an
  // error the user has to act on.
  return error.status == null || error.status >= 500
}

function addIsoDays(value, days) {
  const date = new Date(`${value}T12:00:00Z`)
  date.setUTCDate(date.getUTCDate() + days)
  return date.toISOString().slice(0, 10)
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 8_000) {
  const controller = new AbortController()
  const abortFromCaller = () => controller.abort()
  options.signal?.addEventListener('abort', abortFromCaller, { once: true })
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } finally {
    window.clearTimeout(timeout)
    options.signal?.removeEventListener('abort', abortFromCaller)
  }
}

function snapshotFallback(snapshot, date) {
  const today = new Intl.DateTimeFormat('sv-SE', { timeZone: 'America/New_York' }).format(new Date())
  const forecasts = (snapshot.forecasts ?? []).map((forecast) => ({
    ...forecast, targetDate: date, isCalibrated: false, dataQualityStatus: 'PACKAGED_SNAPSHOT_FALLBACK',
    currentObservedTemperatureF: null, observedHighSoFarF: null, observedLowSoFarF: null, intradayObservations: [], lastObservationAt: null, dataFreshness: null, sourceAgreement: null,
    reasonCodes: ['The live forecast service and direct guidance source are unavailable. These are the last packaged reference values, retained only so the board remains readable.'], stabilityReason: 'packaged_snapshot_fallback',
  }))
  return { ...snapshot, today, targetDate: date, maxDate: addIsoDays(today, 7), generatedAt: snapshot.generatedAt, forecasts, unavailableStations: [], marketForecast: false, forecastInputs: 'Last packaged forecast reference; live guidance is unavailable.', modelStatus: 'PACKAGED_SNAPSHOT_FALLBACK: these values are not a live forecast.', releaseStatus: 'Live forecast refresh is unavailable. Do not use this fallback for a weather pick.' }
}

// The Worker is the normal source of truth. This client fallback is deliberately
// narrow: it only prevents a complete blank board if its upstream proxy is
// unavailable, and identifies the returned values as direct NBM guidance.
async function getDirectGuidanceFallback(date, signal) {
  const snapshotResponse = await fetch(`${API_BASE_URL}/dashboard-snapshot.json`, { cache: 'no-store', signal })
  if (!snapshotResponse.ok) throw new Error('Forecast fallback snapshot is unavailable.')
  const snapshot = await snapshotResponse.json()
  const stations = snapshot.stationRegistry
  if (!Array.isArray(stations) || !stations.length) throw new Error('Forecast station registry is unavailable.')
  const endpoint = new URL('https://api.open-meteo.com/v1/forecast')
  endpoint.searchParams.set('latitude', stations.map((station) => station.latitude).join(','))
  endpoint.searchParams.set('longitude', stations.map((station) => station.longitude).join(','))
  endpoint.searchParams.set('daily', 'temperature_2m_max')
  endpoint.searchParams.set('temperature_unit', 'fahrenheit')
  endpoint.searchParams.set('timezone', 'auto')
  endpoint.searchParams.set('past_days', '1')
  endpoint.searchParams.set('forecast_days', '16')
  endpoint.searchParams.set('models', 'ncep_nbm_conus')
  const guidanceResponse = await fetchWithTimeout(endpoint, { cache: 'no-store', signal })
  if (!guidanceResponse.ok) throw new Error('Live NBM guidance is temporarily unavailable.')
  const guidance = await guidanceResponse.json()
  const items = Array.isArray(guidance) ? guidance : [guidance]
  const priorByStation = new Map((snapshot.forecasts ?? []).map((forecast) => [forecast.station, forecast]))
  const forecasts = stations.map((station, index) => {
    const daily = items[index]?.daily
    const dayIndex = daily?.time?.indexOf(date) ?? -1
    const value = Number(daily?.temperature_2m_max?.[dayIndex])
    if (!Number.isFinite(value)) return null
    const high = Math.round(value)
    return {
      ...(priorByStation.get(station.stationId) ?? {}), station: station.stationId, city: station.name, marketLocation: station.display_name, targetDate: date,
      highF: high, rawModelHighF: high, baselineHighF: high, modelDeltaF: 0, rangeLowF: high - 4, rangeHighF: high + 4, fourDegreeRangeLowF: high - 2, fourDegreeRangeHighF: high + 2, modelRange: [high - 4, high + 4],
      uncertainty: 'High', isCalibrated: false, currentObservedTemperatureF: null, observedHighSoFarF: null, observedLowSoFarF: null, intradayObservations: [], lastObservationAt: null, dataFreshness: null, sourceAgreement: null, sourceCount: 1,
      dataQualityStatus: 'BROWSER_DIRECT_GUIDANCE_FALLBACK', reasonCodes: ['Live NCEP NBM daily guidance loaded directly because the forecast service is unavailable.'], stabilityReason: 'direct_guidance_fallback',
    }
  }).filter(Boolean)
  if (!forecasts.length) throw new Error('Live NBM guidance returned no usable daily highs.')
  const today = new Intl.DateTimeFormat('sv-SE', { timeZone: 'America/New_York' }).format(new Date())
  return { ...snapshot, today, targetDate: date, maxDate: addIsoDays(today, 7), generatedAt: new Date().toISOString(), forecasts, unavailableStations: stations.filter((station) => !forecasts.some((forecast) => forecast.station === station.stationId)).map((station) => station.stationId), marketForecast: true, forecastInputs: 'Live NCEP NBM daily-high guidance via Open-Meteo browser fallback.', modelStatus: 'DIRECT_GUIDANCE_FALLBACK: live NCEP NBM shown without residual calibration.', releaseStatus: 'Direct guidance fallback is active because the forecast service is unavailable.' }
}

export async function getDashboard(date, { signal, force = false } = {}) {
  try {
    return await requestDashboard(date, force, signal)
  } catch (error) {
    if (signal?.aborted || error.name === 'AbortError' || !isTransientFailure(error)) throw error
    await new Promise((resolve) => window.setTimeout(resolve, 1_000))
    if (signal?.aborted) throw error
    try {
      return await requestDashboard(date, force, signal)
    } catch (retryError) {
      if (signal?.aborted || retryError.name === 'AbortError' || !isTransientFailure(retryError)) throw retryError
      try {
        return await getDirectGuidanceFallback(date, signal)
      } catch {
        const snapshotResponse = await fetch(`${API_BASE_URL}/dashboard-snapshot.json`, { cache: 'no-store', signal })
        if (!snapshotResponse.ok) throw retryError
        return snapshotFallback(await snapshotResponse.json(), date)
      }
    }
  }
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    cache: 'no-store',
  })
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const error = new Error(payload?.error || `Account service returned ${response.status}.`)
    error.status = response.status
    throw error
  }
  return payload
}

export function claimDailyPick(stationId, targetDate) {
  return postJson('/api/access/claim', { stationId, targetDate })
}

export function startCheckout() {
  return postJson('/api/billing/checkout')
}

export function openBillingPortal() {
  return postJson('/api/billing/portal')
}
