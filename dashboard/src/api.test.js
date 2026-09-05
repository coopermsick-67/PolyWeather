import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildDirectGuidanceForecast, finiteTemperature, withoutCalibratedEvidence } from './api.js'

const station = {
  stationId: 'KNYC',
  name: 'New York City',
  display_name: 'Central Park',
  display_note: 'Official station',
  timezone: 'America/New_York',
}

test('direct-guidance fallback rejects missing or non-numeric temperatures instead of coercing them to zero', () => {
  for (const value of [null, undefined, '', '   ', false, []]) {
    const daily = { time: ['2026-09-05'], temperature_2m_max: [value] }
    assert.equal(finiteTemperature(value), null)
    assert.equal(buildDirectGuidanceForecast(station, '2026-09-05', daily), null)
  }
  assert.equal(finiteTemperature('74.6'), 74.6)
})

test('direct-guidance fallback has no calibrated recommendation or stale evidence fields', () => {
  const daily = { time: ['2026-09-05'], temperature_2m_max: ['74.6'] }

  const forecast = buildDirectGuidanceForecast(station, '2026-09-05', daily)

  assert.equal(forecast.highF, 75)
  assert.equal(forecast.isCalibrated, false)
  assert.equal(forecast.betDecision, null)
  assert.equal(forecast.rangeLowF, null)
  assert.equal(forecast.rangeHighF, null)
  assert.deepEqual(forecast.sourceProvenance, {
    provider: 'Open-Meteo',
    sourceRunAgeVerified: false,
  })
})

test('fallback dashboard strips calibrated board-level evidence from its snapshot', () => {
  const fallback = withoutCalibratedEvidence({
    accuracy: [{ station: 'KNYC' }],
    modelEvidence: { candidateMaeF: 1.7 },
    trend: [{ date: '2026-09-05' }],
    betSummary: { recommended: 1 },
  })

  assert.deepEqual(fallback.accuracy, [])
  assert.equal(fallback.modelEvidence, null)
  assert.deepEqual(fallback.trend, [])
  assert.equal(fallback.betSummary, null)
})
