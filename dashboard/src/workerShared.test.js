import { test } from 'node:test'
import assert from 'node:assert/strict'
import { addDays, constantTimeEqual, earliestStationLocalToday, isIsoDate, localDate, targetDateForRequest } from './workerShared.js'

test('isIsoDate accepts only real calendar dates', () => {
  assert.equal(isIsoDate('2026-09-03'), true)
  assert.equal(isIsoDate('2026-02-30'), false)
  assert.equal(isIsoDate('not-a-date'), false)
  assert.equal(isIsoDate(''), false)
})

test('targetDateForRequest defaults to today when no value is given', () => {
  assert.equal(targetDateForRequest(undefined, '2026-09-03'), '2026-09-03')
})

test('targetDateForRequest accepts a date within the earliest..today+7 window', () => {
  assert.equal(targetDateForRequest('2026-09-05', '2026-09-03', '2026-09-02'), '2026-09-05')
})

test('targetDateForRequest accepts a still-current Pacific "today" the day before Eastern rolled over', () => {
  // 9:30 PM Pacific = 12:30 AM Eastern the next day: Eastern's "today" has
  // already advanced, but earliestToday (Pacific) has not.
  const eastern = '2026-09-04'
  const pacific = '2026-09-03'
  assert.equal(targetDateForRequest(pacific, eastern, pacific), pacific)
})

test('targetDateForRequest rejects a date before the earliest station-local today', () => {
  assert.throws(() => targetDateForRequest('2026-09-01', '2026-09-03', '2026-09-02'), /Forecast date must be between/)
})

test('targetDateForRequest rejects a date past the 7-day window', () => {
  assert.throws(() => targetDateForRequest('2026-09-20', '2026-09-03'), /Forecast date must be between/)
})

test('targetDateForRequest thrown errors carry a 400 status', () => {
  try {
    targetDateForRequest('garbage', '2026-09-03')
    assert.fail('expected targetDateForRequest to throw')
  } catch (error) {
    assert.equal(error.status, 400)
  }
})

test('earliestStationLocalToday picks the earliest local calendar day across timezones', () => {
  // 2026-09-04T05:00:00Z: New York (EDT, UTC-4) has already rolled over to
  // Sept 4 (01:00 local), but Los Angeles (PDT, UTC-7) is still Sept 3
  // (22:00 local the previous evening).
  const now = new Date('2026-09-04T05:00:00Z')
  const stations = [{ timezone: 'America/New_York' }, { timezone: 'America/Los_Angeles' }]
  assert.equal(localDate(now, 'America/New_York'), '2026-09-04')
  assert.equal(localDate(now, 'America/Los_Angeles'), '2026-09-03')
  assert.equal(earliestStationLocalToday(now, stations), '2026-09-03')
})

test('earliestStationLocalToday returns null for an empty station list', () => {
  assert.equal(earliestStationLocalToday(new Date(), []), null)
})

test('addDays crosses month and year boundaries', () => {
  assert.equal(addDays('2026-12-30', 3), '2027-01-02')
})

test('localDate renders a date in the given IANA timezone', () => {
  // 2026-09-04T12:00:00Z is unambiguously Sept 4 everywhere in the US.
  assert.equal(localDate(new Date('2026-09-04T12:00:00Z'), 'America/Los_Angeles'), '2026-09-04')
  assert.equal(localDate(new Date('2026-09-04T12:00:00Z'), 'America/New_York'), '2026-09-04')
  // 03:30 UTC is still Sept 3 in both US zones (EDT is UTC-4, PDT is UTC-7).
  assert.equal(localDate(new Date('2026-09-04T03:30:00Z'), 'America/New_York'), '2026-09-03')
  assert.equal(localDate(new Date('2026-09-04T03:30:00Z'), 'America/Los_Angeles'), '2026-09-03')
})

test('constantTimeEqual matches equal strings and rejects unequal or mismatched-length ones', () => {
  assert.equal(constantTimeEqual('abc123', 'abc123'), true)
  assert.equal(constantTimeEqual('abc123', 'abc124'), false)
  assert.equal(constantTimeEqual('abc123', 'abc12'), false)
  assert.equal(constantTimeEqual('', ''), true)
})
