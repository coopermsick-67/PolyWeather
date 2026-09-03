import { test } from 'node:test'
import assert from 'node:assert/strict'
import { addDays, dateChoice, formatIso, isoToday, longDate, shortDate, stationDayBounds, stationIsoDate, stationTime } from './dateUtils.js'

test('formatIso pads month and day', () => {
  assert.equal(formatIso(new Date(2026, 0, 5)), '2026-01-05')
  assert.equal(formatIso(new Date(2026, 10, 23)), '2026-11-23')
})

test('isoToday matches formatIso of the current local date', () => {
  const now = new Date()
  assert.equal(isoToday(), formatIso(new Date(now.getFullYear(), now.getMonth(), now.getDate())))
})

test('addDays moves forward across a month boundary', () => {
  assert.equal(addDays('2026-01-30', 3), '2026-02-02')
})

test('addDays moves backward', () => {
  assert.equal(addDays('2026-03-01', -1), '2026-02-28')
})

test('addDays(0) is a no-op', () => {
  assert.equal(addDays('2026-06-15', 0), '2026-06-15')
})

test('dateChoice labels today and tomorrow relative to the given reference', () => {
  assert.equal(dateChoice('2026-06-15', '2026-06-15'), 'Today')
  assert.equal(dateChoice('2026-06-16', '2026-06-15'), 'Tomorrow')
})

test('dateChoice falls back to a weekday/day label further out', () => {
  const label = dateChoice('2026-06-20', '2026-06-15')
  assert.match(label, /\d+/)
  assert.notEqual(label, 'Today')
  assert.notEqual(label, 'Tomorrow')
})

test('longDate and shortDate produce non-empty human labels', () => {
  assert.ok(longDate('2026-06-15').length > 0)
  assert.ok(shortDate('2026-06-15').length > 0)
})

test('station helpers respect the station timezone and DST calendar day', () => {
  assert.equal(stationIsoDate('2026-09-03T03:30:00Z', 'America/Los_Angeles'), '2026-09-02')
  assert.equal(stationTime('2026-09-03T03:30:00Z', 'America/Los_Angeles'), '8:30 PM')
  const bounds = stationDayBounds('2026-03-08', 'America/New_York')
  assert.equal(bounds.end - bounds.start, 23 * 60 * 60 * 1000)
})
