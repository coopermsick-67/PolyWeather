const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export async function getDashboard(date, { signal } = {}) {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 30_000)
  const abortFromCaller = () => controller.abort()
  signal?.addEventListener('abort', abortFromCaller, { once: true })
  try {
    const url = `${API_BASE_URL}/api/dashboard?date=${encodeURIComponent(date)}`
    const response = await fetch(url, { signal: controller.signal })
    const contentType = response.headers.get('content-type') || ''
    const payload = contentType.includes('application/json') ? await response.json() : null
    if (!response.ok) throw new Error(payload?.error || `Forecast service returned ${response.status}.`)
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
