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

export async function getDashboard(date, { signal, force = false } = {}) {
  try {
    return await requestDashboard(date, force, signal)
  } catch (error) {
    if (signal?.aborted || error.name === 'AbortError' || !isTransientFailure(error)) throw error
    await new Promise((resolve) => window.setTimeout(resolve, 1_000))
    if (signal?.aborted) throw error
    return requestDashboard(date, force, signal)
  }
}
