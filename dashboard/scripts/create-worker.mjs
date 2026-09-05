import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'

const dist = new URL('../dist/', import.meta.url)
const publicDir = new URL('../dist/public/', import.meta.url)
const entries = (await readdir(dist, { withFileTypes: true })).filter((entry) => entry.name !== 'server' && entry.name !== 'public')

await rm(publicDir, { recursive: true, force: true })
await mkdir(publicDir, { recursive: true })
for (const entry of entries) {
  await cp(new URL(`../dist/${entry.name}`, import.meta.url), new URL(`../dist/public/${entry.name}`, import.meta.url), { recursive: entry.isDirectory() })
}

const assetNames = [
  'index.html',
  'dashboard-snapshot.json',
  'backtest-data.json',
  'backtest-worker.js',
  ...(await readdir(new URL('../dist/assets/', import.meta.url))).map((name) => `assets/${name}`),
]
const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
}
const assets = Object.fromEntries(await Promise.all(assetNames.map(async (name) => [
  `/${name}`,
  {
    body: (await readFile(new URL(`../dist/${name}`, import.meta.url))).toString('base64'),
    type: contentTypes[`.${name.split('.').pop()}`] ?? 'application/octet-stream',
  },
])))
// dashboard/src/workerShared.js is the tested (node --test) source of truth
// for these date/security helpers. Splicing its text in here -- rather than
// hand-duplicating the logic in this template -- means the code exercised
// by tests is exactly the code that ships in the deployed Worker, not a
// copy that can silently drift from it.
const sharedHelpersSource = (await readFile(new URL('../src/workerShared.js', import.meta.url), 'utf8'))
  .replace(/^export (function|const)/gm, '$1')
const worker = `const assets = ${JSON.stringify(assets)};
const decode = (value) => Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
const snapshot = JSON.parse(new TextDecoder().decode(decode(assets['/dashboard-snapshot.json'].body)));
const responseCache = new Map();
const refreshRequests = new Map();
const CACHE_TTL_MS = 15 * 60 * 1000;
const FORCE_REFRESH_COOLDOWN_MS = 60 * 1000;
// Render's free tier spins the API down after inactivity; a cold start plus
// live NWS/Open-Meteo fetches can legitimately take 30-50s. This is set just
// under server.py's own gunicorn --timeout 60 so a slow-but-alive backend
// response wins the race instead of the Worker giving up first.
const FORECAST_API_TIMEOUT_MS = 55_000;
// The single authoritative forecast implementation: server.py / dashboard_
// payload.py (real joblib/scikit-learn/XGBoost inference, tested by
// tests/test_dashboard_payload.py and tests/test_server.py). This Worker no
// longer re-derives features or re-implements the model by hand in
// JavaScript -- that used to be a second, independently-drifting copy of
// the forecast pipeline (it had, among other bugs, a hardcoded tree
// missing-value routing that was the exact opposite of what the deployed
// model actually needed on every single node). The Worker is now a thin
// proxy that adds auth/paywall/billing in front of the one real backend.
// Override via FORECAST_API_URL for a non-default deployment (e.g. staging).
const DEFAULT_FORECAST_API_URL = 'https://polyweather-api.onrender.com';

${sharedHelpersSource}

async function fetchWithTimeout(input, init = {}, timeoutMs = FORECAST_API_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

// Cloudflare Workers limit concurrent outbound responses.  Always consume or
// cancel an unsuccessful body so a retry cannot leave a response in-flight.
async function jsonIfOk(response) {
  if (!response?.ok) {
    try { await response?.body?.cancel(); } catch { /* Nothing to release. */ }
    return null;
  }
  return response.json().catch(() => null);
}

async function fetchFlaskDashboard(targetDate, env) {
  const apiUrl = env.FORECAST_API_URL || DEFAULT_FORECAST_API_URL;
  const endpoint = new URL('/api/dashboard', apiUrl);
  if (targetDate) endpoint.searchParams.set('date', targetDate);
  let response;
  try {
    response = await fetchWithTimeout(endpoint, { headers: { Accept: 'application/json' } });
  } catch (error) {
    throw new Error('Forecast service is unreachable: ' + (error.message || 'network error'));
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error || 'Forecast service returned ' + response.status + '.');
  if (!body || !Array.isArray(body.forecasts)) throw new Error('Forecast service returned an invalid response.');
  return body;
}

function directNbmEndpoint(stations) {
  const endpoint = new URL('https://api.open-meteo.com/v1/forecast');
  for (const [key, value] of Object.entries({
    latitude: stations.map((station) => station.latitude).join(','),
    longitude: stations.map((station) => station.longitude).join(','),
    daily: 'temperature_2m_max',
    temperature_unit: 'fahrenheit',
    timezone: 'auto',
    forecast_days: '16',
    // At UTC midnight, several settlement stations are still on their
    // previous local calendar day. Asking for one prior day guarantees the
    // station-local "today" index exists during that handoff instead of
    // returning an empty board until the next upstream model cycle.
    past_days: '1',
    models: 'ncep_nbm_conus',
  })) endpoint.searchParams.set(key, value);
  return endpoint;
}

// Fallback guidance is useful weather information, but it is not the
// calibrated model represented by a packaged dashboard snapshot. Remove all
// board-level calibrated evidence along with per-station decisions.
function fallbackSnapshotBase() {
  return {
    ...snapshot,
    accuracy: [],
    modelEvidence: null,
    trend: [],
    betSummary: null,
  };
}

// Number(x) coerces null to 0 and undefined/non-numeric strings to NaN. A
// bare Number.isFinite(Number(x)) check therefore treats a genuine null
// temperature value (which Open-Meteo can return for a day outside a
// model's actual coverage) as a valid, finite 0F reading instead of "no
// data." Reject null/undefined explicitly before coercing.
function finiteTemperature(value) {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function directGuidanceForecast(station, targetDate, item, model = 'ncep_nbm_conus') {
  const daily = item?.daily;
  const index = Array.isArray(daily?.time) ? daily.time.indexOf(targetDate) : -1;
  const value = finiteTemperature(daily?.temperature_2m_max?.[index]);
  if (value === null) return null;
  const high = Math.round(value);
  const isBestMatch = model === 'best_match';
  const sourceName = isBestMatch
    ? 'Live Open-Meteo best-match guidance (requested NBM model was unavailable)'
    : 'Live NCEP NBM via Open-Meteo';
  const reasonCode = isBestMatch
    ? 'Requested NCEP NBM guidance was unavailable for this station; unspecified Open-Meteo best-match guidance is shown instead. The calibrated forecast service is unavailable, so residual MOS and observed-high adjustments are deliberately disabled.'
    : 'Live NCEP NBM daily guidance via Open-Meteo. The calibrated forecast service is unavailable, so residual MOS and observed-high adjustments are deliberately disabled.';
  // A prior snapshot (packaged reference or a previous live/calibrated
  // response) is intentionally NOT spread into this object: doing so used to
  // let unlisted fields -- most importantly betDecision, a real
  // recommendation computed against a *different* forecast -- leak through
  // unnoticed onto this uncalibrated fallback result. Every field this
  // fallback can honestly populate is listed explicitly below instead.
  return {
    station: station.stationId,
    city: station.name,
    marketLocation: station.display_name,
    settlementNote: station.display_note,
    settlementSource: null,
    timezone: station.timezone,
    marketType: 'daily_high',
    targetDate,
    highF: high,
    rawModelHighF: high,
    baselineHighF: high,
    modelDeltaF: 0,
    rangeLowF: null,
    rangeHighF: null,
    fourDegreeRangeLowF: null,
    fourDegreeRangeHighF: null,
    modelRange: null,
    uncertainty: 'Unavailable',
    isCalibrated: false,
    forecastLeadDays: null,
    evaluatedLeadDays: null,
    supportedHorizon: false,
    featureCompletenessPct: null,
    guidanceComplete: false,
    requiredGuidanceModels: [],
    sourceProvenance: null,
    currentObservedTemperatureF: null,
    observedHighSoFarF: null,
    observedLowSoFarF: null,
    intradayObservations: [],
    lastObservationAt: null,
    dataFreshness: null,
    sourceAgreement: null,
    modelSpreadF: null,
    sourceCount: 1,
    sourceName,
    dataQualityStatus: 'DIRECT_GUIDANCE_FALLBACK',
    reasonCodes: [reasonCode],
    stabilityReason: 'direct_guidance_fallback',
    // No calibrated distribution exists for a direct-guidance fallback, so
    // there is nothing honest to recommend a bet against.
    betDecision: null,
  };
}

async function directNbmDashboard(targetDate) {
  const now = new Date();
  const today = localDate(now, 'America/New_York');
  const resolvedDate = targetDateForRequest(targetDate, today, earliestStationLocalToday(now, snapshot.stationRegistry));
  let response;
  let payload;
  try {
    response = await fetchWithTimeout(directNbmEndpoint(snapshot.stationRegistry), { headers: { Accept: 'application/json' } }, 15_000);
    payload = await jsonIfOk(response);
  } catch (error) {
    console.warn('Bulk NBM fetch failed; retrying each station.', error.message || error);
  }
  // Open-Meteo occasionally rejects a multi-location request at the edge.
  // Retry independent station requests, and only then use best_match as a
  // clearly labeled availability fallback. One city must not blank all 20.
  let stations = Array.isArray(payload) ? payload : [payload];
  // Every station in the initial bulk request was asked for ncep_nbm_conus
  // specifically; only a per-station repair below may substitute best_match.
  let stationModels = snapshot.stationRegistry.map(() => 'ncep_nbm_conus');
  const hasRequestedHigh = (item) => {
    const daily = item?.daily;
    const index = Array.isArray(daily?.time) ? daily.time.indexOf(resolvedDate) : -1;
    return finiteTemperature(daily?.temperature_2m_max?.[index]) !== null;
  };
  if (!payload || stations.length !== snapshot.stationRegistry.length || stations.some((item) => !hasRequestedHigh(item))) {
    const missing = snapshot.stationRegistry.map((station, index) => ({ station, index })).filter(({ index }) => !hasRequestedHigh(stations[index]));
    // A 20-request Promise.all leaves Cloudflare with stalled response bodies
    // when an upstream model is slow. Repair at most four stations at once;
    // this preserves the all-city fallback without exhausting worker fetches.
    for (let offset = 0; offset < missing.length; offset += 4) {
      const repaired = await Promise.all(missing.slice(offset, offset + 4).map(async ({ station, index }) => {
      const requestFor = async (model) => {
        const endpoint = directNbmEndpoint([station]);
        endpoint.searchParams.set('models', model);
        const point = await fetchWithTimeout(endpoint, { headers: { Accept: 'application/json' } }, 8_000);
        const item = await jsonIfOk(point);
        return item ? { item, model } : null;
      };
        try {
          const result = (await requestFor('ncep_nbm_conus')) || (await requestFor('best_match'));
          return { index, item: result?.item ?? null, model: result?.model ?? null };
        } catch { return { index, item: null, model: null }; }
      }));
      for (const { index, item, model } of repaired) {
        stations[index] = item;
        // A repaired station that still failed keeps its prior (unused)
        // label; only a successful repair may relabel it as best_match.
        if (model) stationModels[index] = model;
      }
    }
  }
  const forecasts = snapshot.stationRegistry
    .map((station, index) => directGuidanceForecast(station, resolvedDate, stations[index], stationModels[index]))
    .filter(Boolean);
  if (!forecasts.length) throw new Error('Direct NCEP NBM guidance returned no usable daily highs.');
  const available = new Set(forecasts.map((forecast) => forecast.station));
  return {
    ...fallbackSnapshotBase(),
    today,
    targetDate: resolvedDate,
    maxDate: addDays(today, 7),
    generatedAt: now.toISOString(),
    forecasts,
    unavailableStations: snapshot.stationRegistry.filter((station) => !available.has(station.stationId)).map((station) => station.stationId),
    marketForecast: true,
    // The packaged snapshot can contain an old board-level recommendation
    // summary. This fallback has no calibrated decisions, so it cannot carry
    // that summary forward even though individual forecast objects are safe.
    forecastInputs: 'Live NCEP NBM daily-high guidance via Open-Meteo. Calibrated server-side residual MOS is temporarily unavailable.',
    modelStatus: 'DIRECT_GUIDANCE_FALLBACK: live NCEP NBM shown without residual calibration.',
    releaseStatus: 'Fallback guidance refreshes every 15 minutes. It is live, but it is not a substitute for the calibrated 20-station forecast service.',
  };
}

const NWS_HEADERS = {
  Accept: 'application/geo+json',
  // NWS asks API clients to identify themselves. This stable product name is
  // also preferable to forwarding a visitor's browser user agent upstream.
  'User-Agent': 'WeatherPicks live dashboard (weatherpicks.coopdogg67.chatgpt.site)',
};

function fahrenheit(celsius) {
  return Number.isFinite(celsius) ? celsius * 9 / 5 + 32 : null;
}

async function nwsJson(url) {
  const response = await fetchWithTimeout(url, { headers: NWS_HEADERS }, 10_000);
  const value = await jsonIfOk(response);
  if (!value) throw new Error('NWS did not return usable live weather data.');
  return value;
}

function dateAtStation(value, station) {
  return localDate(new Date(value), station.timezone);
}

async function nwsStationForecast(station, targetDate, today) {
  const forecastUrl = 'https://api.weather.gov/gridpoints/' + station.forecastGrid.office + '/' + station.forecastGrid.x + ',' + station.forecastGrid.y + '/forecast';
  const observationUrl = 'https://api.weather.gov/stations/' + station.stationId + '/observations?limit=100';
  const [forecastPayload, observationsPayload] = await Promise.all([
    nwsJson(forecastUrl),
    targetDate === today ? nwsJson(observationUrl).catch(() => null) : Promise.resolve(null),
  ]);
  const period = (forecastPayload.properties?.periods ?? []).find((item) =>
    item.isDaytime && dateAtStation(item.startTime, station) === targetDate && finiteTemperature(item.temperature) !== null
  );
  if (!period) return null;

  const observations = (observationsPayload?.features ?? [])
    .map((feature) => {
      const properties = feature.properties ?? {};
      return { time: properties.timestamp, temperatureF: fahrenheit(properties.temperature?.value) };
    })
    .filter((item) => item.time && Number.isFinite(item.temperatureF) && dateAtStation(item.time, station) === targetDate)
    .sort((left, right) => new Date(left.time) - new Date(right.time));
  const latest = observations.at(-1) ?? null;
  const observedTemperatures = observations.map((item) => item.temperatureF);
  const observedHigh = observedTemperatures.length ? Math.max(...observedTemperatures) : null;
  const observedLow = observedTemperatures.length ? Math.min(...observedTemperatures) : null;
  const nwsHigh = Math.round(finiteTemperature(period.temperature));
  // A live reported station high is factual and must never be overwritten by
  // a forecast that was issued before that observation arrived.
  const high = Math.max(nwsHigh, observedHigh == null ? -Infinity : Math.round(observedHigh));
  // A prior snapshot is intentionally NOT spread into this object -- see the
  // matching comment in directGuidanceForecast. Every field this fallback
  // can honestly populate is listed explicitly below instead.
  return {
    station: station.stationId,
    city: station.name,
    marketLocation: station.display_name,
    settlementNote: station.display_note,
    settlementSource: null,
    timezone: station.timezone,
    marketType: 'daily_high',
    targetDate,
    highF: high,
    rawModelHighF: nwsHigh,
    baselineHighF: nwsHigh,
    modelDeltaF: high - nwsHigh,
    rangeLowF: null,
    rangeHighF: null,
    fourDegreeRangeLowF: null,
    fourDegreeRangeHighF: null,
    modelRange: null,
    uncertainty: 'Unavailable',
    isCalibrated: false,
    forecastLeadDays: null,
    evaluatedLeadDays: null,
    supportedHorizon: false,
    featureCompletenessPct: null,
    guidanceComplete: false,
    requiredGuidanceModels: [],
    sourceProvenance: {
      provider: 'National Weather Service',
      fetchedAt: new Date().toISOString(),
      publishedAt: forecastPayload.properties?.updated ?? null,
      // NWS publication time is not an underlying numerical-model run ID,
      // so it cannot establish the age of the source guidance.
      sourceRunAgeVerified: false,
    },
    currentObservedTemperatureF: latest ? Math.round(latest.temperatureF * 10) / 10 : null,
    observedHighSoFarF: observedHigh == null ? null : Math.round(observedHigh),
    observedLowSoFarF: observedLow == null ? null : Math.round(observedLow),
    intradayObservations: observations,
    lastObservationAt: latest?.time ?? null,
    dataFreshness: null,
    sourceAgreement: null,
    modelSpreadF: null,
    sourceName: 'NWS official daily forecast',
    sourceCount: 1,
    dataQualityStatus: 'LIVE_NWS_FORECAST',
    reasonCodes: ['NWS daily forecast: ' + (period.shortForecast || ('high near ' + nwsHigh + '°F')) + '. ' + (latest ? 'Station observations are live.' : 'No station observation has been reported for this selected day yet.')],
    stabilityReason: observedHigh != null && high > nwsHigh ? 'observed_high' : 'nws_live_forecast',
    // No calibrated distribution exists for a live NWS fallback, so there is
    // nothing honest to recommend a bet against.
    betDecision: null,
  };
}

async function nwsLiveDashboard(targetDate) {
  const now = new Date();
  const today = localDate(now, 'America/New_York');
  const resolvedDate = targetDateForRequest(targetDate, today, earliestStationLocalToday(now, snapshot.stationRegistry));
  const forecasts = [];
  // NWS provides one forecast-grid and, for today, one station-observation
  // request per market. Keep only four in flight so this stays well within
  // the Worker's outbound-request budget and remains responsive.
  for (let offset = 0; offset < snapshot.stationRegistry.length; offset += 4) {
    const batch = await Promise.all(snapshot.stationRegistry.slice(offset, offset + 4).map(async (station) => {
      try { return await nwsStationForecast(station, resolvedDate, today); } catch { return null; }
    }));
    forecasts.push(...batch.filter(Boolean));
  }
  if (!forecasts.length) throw new Error('NWS did not return daily forecasts for any configured station.');
  const available = new Set(forecasts.map((forecast) => forecast.station));
  return {
    ...fallbackSnapshotBase(),
    today,
    targetDate: resolvedDate,
    maxDate: addDays(today, 7),
    generatedAt: now.toISOString(),
    forecasts,
    unavailableStations: snapshot.stationRegistry.filter((station) => !available.has(station.stationId)).map((station) => station.stationId),
    marketForecast: true,
    forecastInputs: 'Live NWS daily forecasts and station observations.',
    modelStatus: 'LIVE_NWS_FORECAST: official live guidance; no residual calibration is applied.',
    releaseStatus: 'Live NWS forecast and station observations refresh every 15 minutes.',
  };
}

// If every live source is unavailable, fail visibly.  A stale packaged
// temperature can look like a real prediction even with a warning beside it.
function packagedSnapshotDashboard(targetDate) {
  const now = new Date();
  const today = localDate(now, 'America/New_York');
  const resolvedDate = targetDateForRequest(targetDate, today, earliestStationLocalToday(now, snapshot.stationRegistry));
  return {
    ...fallbackSnapshotBase(),
    today,
    targetDate: resolvedDate,
    maxDate: addDays(today, 7),
    generatedAt: null,
    forecasts: [],
    unavailableStations: snapshot.stationRegistry.map((station) => station.stationId),
    marketForecast: false,
    forecastInputs: 'Live guidance is unavailable; no cached temperatures are substituted.',
    modelStatus: 'LIVE_DATA_UNAVAILABLE',
    releaseStatus: 'Live forecast refresh is unavailable. No stale or synthetic forecast values are displayed.',
  };
}

async function liveDashboard(targetDate, env) {
  try {
    const dashboard = await fetchFlaskDashboard(targetDate, env);
    const registry = Array.isArray(dashboard.stationRegistry) && dashboard.stationRegistry.length ? dashboard.stationRegistry : snapshot.stationRegistry;
    const forecastStations = new Set(dashboard.forecasts.map((forecast) => forecast.station));
    // dashboard_payload.py omits a station entirely (rather than fabricating a
    // value) when its live inputs are incomplete; surface which ones so the
    // frontend can say so instead of the station silently vanishing.
    const unavailableStations = registry.filter((station) => !forecastStations.has(station.stationId)).map((station) => station.stationId);
    return { ...dashboard, stationRegistry: registry, unavailableStations, marketForecast: true };
  } catch (error) {
    console.warn('Calibrated forecast API failed; serving labeled direct-guidance fallback.', error.message || error);
    try {
      return await nwsLiveDashboard(targetDate);
    } catch (nwsError) {
      console.warn('Live NWS fallback failed; trying direct NBM guidance.', nwsError.message || nwsError);
      try {
        return await directNbmDashboard(targetDate);
      } catch (fallbackError) {
        console.warn('Direct guidance fallback failed; serving packaged reference board.', fallbackError.message || fallbackError);
        return packagedSnapshotDashboard(targetDate);
      }
    }
  }
}

function dashboardHeaders() {
  return { 'cache-control': 'private, no-store' };
}

function accountHeaders() {
  return { 'cache-control': 'no-store' };
}

function jsonError(message, status = 400) {
  return Response.json({ error: message }, { status, headers: accountHeaders() });
}

function identityFrom(request, env) {
  // oai-authenticated-user-id/-email are meant to be set by a trusted
  // upstream gateway (ChatGPT Apps) after it authenticates the caller. This
  // Worker also serves the public SPA and API at its own directly-reachable
  // URL, and nothing else here verifies these headers actually came from
  // that trusted gateway rather than being forged by any HTTP client hitting
  // the Worker directly -- which would let anyone impersonate any user,
  // including an admin (by setting the admin's email), bypassing the
  // paywall entirely. When IDENTITY_GATEWAY_SECRET is configured as a
  // Worker secret, require a matching x-weatherpicks-gateway-secret header
  // (set by the trusted gateway on every forwarded request) before trusting
  // the identity headers at all.
  const gatewaySecret = env.IDENTITY_GATEWAY_SECRET;
  if (gatewaySecret) {
    const provided = request.headers.get('x-weatherpicks-gateway-secret') || '';
    if (!constantTimeEqual(provided, gatewaySecret)) return null;
  } else {
    console.error('SECURITY WARNING: IDENTITY_GATEWAY_SECRET is not configured. oai-authenticated-user-* headers are being trusted with no verification that they came from a real trusted gateway -- this Worker is currently vulnerable to full identity spoofing. Configure IDENTITY_GATEWAY_SECRET as a Worker secret and have the trusted gateway attach it as x-weatherpicks-gateway-secret to close this gap.');
  }
  const id = request.headers.get('oai-authenticated-user-id');
  const email = request.headers.get('oai-authenticated-user-email');
  return id && email ? { id, email: email.trim().toLowerCase() } : null;
}

function todayForAccess() {
  return localDate(new Date(), 'America/New_York');
}

function trialEnd(trialStartedAt) {
  return Number(trialStartedAt) + 7 * 24 * 60 * 60 * 1000;
}

function isPaidStatus(status) {
  return status === 'active' || status === 'trialing';
}

function publicAccount(account, env) {
  if (!account) return { authenticated: false, canViewForecasts: false, signInPath: '/signin-with-chatgpt' };
  const now = Date.now();
  const isAdmin = account.role === 'admin';
  const paid = isPaidStatus(account.subscription_status) && (!account.subscription_period_end || Number(account.subscription_period_end) > now);
  const trialEndsAt = trialEnd(account.trial_started_at);
  const trialActive = !isAdmin && !paid && now < trialEndsAt;
  const tier = isAdmin ? 'admin' : paid ? 'member' : trialActive ? 'free_trial' : 'free_expired';
  return {
    authenticated: true,
    email: account.email,
    role: account.role,
    tier,
    canViewForecasts: isAdmin || paid,
    trialActive,
    trialEndsAt: new Date(trialEndsAt).toISOString(),
    subscriptionStatus: account.subscription_status,
    billingConfigured: Boolean(env.STRIPE_SECRET_KEY && env.STRIPE_WEEKLY_PRICE_ID),
    priceLabel: '$10/week',
  };
}

async function requireDatabase(env) {
  if (!env.DB) throw new Error('Account storage is not configured.');
  return env.DB;
}

async function accountForRequest(request, env) {
  const identity = identityFrom(request, env);
  if (!identity) return null;
  const db = await requireDatabase(env);
  const now = Date.now();
  const adminEmail = String(env.WEATHERPICKS_ADMIN_EMAIL || '').trim().toLowerCase();
  const role = adminEmail && identity.email === adminEmail ? 'admin' : 'member';
  let account = await db.prepare('SELECT id, email, role, trial_started_at, stripe_customer_id, stripe_subscription_id, subscription_status, subscription_period_end, created_at, updated_at FROM users WHERE id = ?').bind(identity.id).first();
  if (!account) {
    await db.prepare('INSERT INTO users (id, email, role, trial_started_at, subscription_status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)').bind(identity.id, identity.email, role, now, 'none', now, now).run();
  } else if (account.email !== identity.email || account.role !== role) {
    await db.prepare('UPDATE users SET email = ?, role = ?, updated_at = ? WHERE id = ?').bind(identity.email, role, now, identity.id).run();
  }
  account = await db.prepare('SELECT id, email, role, trial_started_at, stripe_customer_id, stripe_subscription_id, subscription_status, subscription_period_end, created_at, updated_at FROM users WHERE id = ?').bind(identity.id).first();
  return account;
}

async function existingDailyPick(db, userId, accessDate) {
  return db.prepare('SELECT id, user_id, access_date, target_date, station_id, claimed_at FROM daily_pick_access WHERE user_id = ? AND access_date = ?').bind(userId, accessDate).first();
}

function dashboardSummary(targetDate, account, dailyPick) {
  const today = todayForAccess();
  const earliestToday = earliestStationLocalToday(new Date(), snapshot.stationRegistry);
  return {
    today,
    targetDate: targetDateForRequest(targetDate, today, earliestToday),
    maxDate: addDays(today, 7),
    stationRegistry: snapshot.stationRegistry,
    forecasts: [],
    accuracy: snapshot.accuracy ?? [],
    account: { ...account, dailyPick: dailyPick ? { stationId: dailyPick.station_id, targetDate: dailyPick.target_date, accessDate: dailyPick.access_date } : null },
    accessMessage: account.authenticated ? 'Choose a station to use today’s free pick.' : 'Sign in to start a free seven-day trial with one daily pick.',
  };
}

async function dashboardValue(url, targetDate, forceRefresh, env) {
  const refreshKey = targetDate ?? 'today';
  const lastRefreshAt = refreshRequests.get(refreshKey) ?? 0;
  const mayForceRefresh = forceRefresh && Date.now() - lastRefreshAt >= FORCE_REFRESH_COOLDOWN_MS;
  if (mayForceRefresh) refreshRequests.set(refreshKey, Date.now());
  const cacheBucket = Math.floor(Date.now() / CACHE_TTL_MS);
  const cacheKey = new Request(url.origin + '/api/dashboard-cache?date=' + encodeURIComponent(targetDate ?? 'today') + '&bucket=' + cacheBucket);
  if (!mayForceRefresh && typeof caches !== 'undefined') {
    try {
      const cachedResponse = await caches.default.match(cacheKey);
      if (cachedResponse) return cachedResponse.json();
    } catch { /* Sites may disable the shared cache; the worker cache still prevents refresh churn. */ }
  }
  const memoryKey = (targetDate ?? 'today') + ':' + cacheBucket;
  const memory = responseCache.get(memoryKey);
  if (!mayForceRefresh && memory && Date.now() - memory.createdAt < CACHE_TTL_MS) return memory.value;
  const value = liveDashboard(targetDate, env);
  responseCache.set(memoryKey, { createdAt: Date.now(), value });
  try {
    const dashboard = await value;
    const response = Response.json(dashboard, { headers: dashboardHeaders() });
    if (typeof caches !== 'undefined') {
      try { await caches.default.put(cacheKey, response.clone()); } catch { /* Use the 15-minute worker cache when shared caching is unavailable. */ }
    }
    return dashboard;
  } catch (error) {
    responseCache.delete(memoryKey);
    throw error;
  }
}

async function dashboardForRequest(request, url, env) {
  const dashboard = await dashboardValue(url, url.searchParams.get('date'), url.searchParams.get('refresh') === '1', env);
  // Forecast research is public. Do not require ChatGPT identity, D1, Stripe,
  // a trial, or a per-station claim just to load the board.
  return { ...dashboard, account: { authenticated: false, canViewForecasts: true, tier: 'public' } };
}

async function claimDailyPick(request, env) {
  const accountRecord = await accountForRequest(request, env);
  if (!accountRecord) return jsonError('Sign in with ChatGPT to claim a daily pick.', 401);
  const account = publicAccount(accountRecord, env);
  if (account.canViewForecasts) return Response.json({ dailyPick: null, account }, { headers: accountHeaders() });
  if (!account.trialActive) return jsonError('Your free trial has ended. Subscribe to unlock the full board.', 403);
  let body;
  try { body = await request.json(); } catch { return jsonError('Choose a valid settlement station.'); }
  const stationId = String(body?.stationId || '');
  const targetDate = String(body?.targetDate || '');
  if (!snapshot.stationRegistry.some((station) => station.stationId === stationId)) return jsonError('Choose a configured settlement station.');
  try { targetDateForRequest(targetDate, todayForAccess(), earliestStationLocalToday(new Date(), snapshot.stationRegistry)); } catch (error) { return jsonError(error.message || 'Choose a valid forecast date.'); }
  const db = await requireDatabase(env);
  const accessDate = todayForAccess();
  const existing = await existingDailyPick(db, accountRecord.id, accessDate);
  if (existing) {
    if (existing.station_id === stationId && existing.target_date === targetDate) return Response.json({ dailyPick: { stationId, targetDate, accessDate }, account }, { headers: accountHeaders() });
    return jsonError('Today’s free pick has already been used.', 409);
  }
  try {
    await db.prepare('INSERT INTO daily_pick_access (id, user_id, access_date, target_date, station_id, claimed_at) VALUES (?, ?, ?, ?, ?, ?)').bind(crypto.randomUUID(), accountRecord.id, accessDate, targetDate, stationId, Date.now()).run();
  } catch {
    return jsonError('Today’s free pick was already claimed. Refresh to see it.', 409);
  }
  return Response.json({ dailyPick: { stationId, targetDate, accessDate }, account }, { headers: accountHeaders() });
}

async function stripeRequest(path, env, form) {
  const response = await fetch('https://api.stripe.com/v1/' + path, { method: 'POST', headers: { Authorization: 'Bearer ' + env.STRIPE_SECRET_KEY, 'Content-Type': 'application/x-www-form-urlencoded' }, body: form.toString() });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || 'Stripe could not create this session.');
  return payload;
}

async function createCheckoutSession(request, url, env) {
  const accountRecord = await accountForRequest(request, env);
  if (!accountRecord) return jsonError('Sign in with ChatGPT before subscribing.', 401);
  const account = publicAccount(accountRecord, env);
  if (account.role === 'admin') return jsonError('Administrator access is already active.', 409);
  if (!env.STRIPE_SECRET_KEY || !env.STRIPE_WEEKLY_PRICE_ID) return jsonError('Subscription checkout is not configured yet.', 503);
  const form = new URLSearchParams({ mode: 'subscription', success_url: url.origin + '/?checkout=success#/account', cancel_url: url.origin + '/?checkout=cancelled#/account', client_reference_id: accountRecord.id, 'line_items[0][price]': env.STRIPE_WEEKLY_PRICE_ID, 'line_items[0][quantity]': '1', 'subscription_data[metadata][user_id]': accountRecord.id });
  if (accountRecord.stripe_customer_id) form.set('customer', accountRecord.stripe_customer_id);
  else form.set('customer_email', accountRecord.email);
  try {
    const session = await stripeRequest('checkout/sessions', env, form);
    return Response.json({ url: session.url }, { headers: accountHeaders() });
  } catch (error) { return jsonError(error.message || 'Subscription checkout could not start.', 502); }
}

async function createBillingPortal(request, url, env) {
  const accountRecord = await accountForRequest(request, env);
  if (!accountRecord) return jsonError('Sign in with ChatGPT before managing billing.', 401);
  if (!env.STRIPE_SECRET_KEY || !accountRecord.stripe_customer_id) return jsonError('Billing management is unavailable for this account.', 409);
  try {
    const session = await stripeRequest('billing_portal/sessions', env, new URLSearchParams({ customer: accountRecord.stripe_customer_id, return_url: url.origin + '/#/account' }));
    return Response.json({ url: session.url }, { headers: accountHeaders() });
  } catch (error) { return jsonError(error.message || 'Billing portal could not start.', 502); }
}

async function verifiedStripeEvent(request, env) {
  const signature = request.headers.get('stripe-signature');
  if (!signature || !env.STRIPE_WEBHOOK_SECRET) return null;
  const values = Object.fromEntries(signature.split(',').map((part) => part.split('=').map((item) => item.trim())).filter((parts) => parts.length === 2));
  const timestamp = Number(values.t);
  if (!Number.isFinite(timestamp) || Math.abs(Date.now() / 1000 - timestamp) > 300 || !values.v1) return null;
  const payload = await request.text();
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(env.STRIPE_WEBHOOK_SECRET), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signatureBytes = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(String(timestamp) + '.' + payload));
  if (!constantTimeEqual(hex(signatureBytes), values.v1)) return null;
  try { return JSON.parse(payload); } catch { return null; }
}

async function processStripeWebhook(request, env) {
  const event = await verifiedStripeEvent(request, env);
  if (!event) return jsonError('Webhook signature verification failed.', 400);
  const db = await requireDatabase(env);
  // Check for prior processing, but do not record it until the state
  // mutation below has actually succeeded. Recording it first (as this used
  // to) meant a failed mutation left a "processed" row behind anyway --
  // Stripe's retry of the same event would then short-circuit on the
  // dedupe check and the lost update would never be retried. Every mutation
  // below is an idempotent SET-to-fixed-value UPDATE, so a rare race
  // between two near-simultaneous deliveries of the same event applying it
  // twice is harmless; only the final INSERT needs to tolerate that race.
  const already = await db.prepare('SELECT event_id FROM stripe_events WHERE event_id = ?').bind(event.id).first();
  if (already) return Response.json({ received: true }, { headers: accountHeaders() });
  const object = event.data?.object ?? {};
  if (event.type === 'checkout.session.completed') {
    const userId = object.client_reference_id || object.metadata?.user_id;
    if (userId) await db.prepare('UPDATE users SET stripe_customer_id = ?, stripe_subscription_id = ?, subscription_status = ?, updated_at = ? WHERE id = ?').bind(object.customer || null, object.subscription || null, 'pending', Date.now(), userId).run();
  }
  if (event.type === 'customer.subscription.created' || event.type === 'customer.subscription.updated' || event.type === 'customer.subscription.deleted') {
    const userId = object.metadata?.user_id;
    const status = event.type === 'customer.subscription.deleted' ? 'canceled' : String(object.status || 'none');
    const periodEnd = Number(object.current_period_end) ? Number(object.current_period_end) * 1000 : null;
    if (userId) await db.prepare('UPDATE users SET stripe_customer_id = ?, stripe_subscription_id = ?, subscription_status = ?, subscription_period_end = ?, updated_at = ? WHERE id = ?').bind(object.customer || null, object.id || null, status, periodEnd, Date.now(), userId).run();
    else if (object.customer) await db.prepare('UPDATE users SET stripe_subscription_id = ?, subscription_status = ?, subscription_period_end = ?, updated_at = ? WHERE stripe_customer_id = ?').bind(object.id || null, status, periodEnd, Date.now(), object.customer).run();
  }
  try {
    await db.prepare('INSERT INTO stripe_events (event_id, event_type, received_at) VALUES (?, ?, ?)').bind(event.id, event.type, Date.now()).run();
  } catch {
    // A concurrent delivery of the same event already recorded it between
    // our check above and here; the mutation we just applied is idempotent,
    // so it is safe to let that other delivery's record stand.
  }
  return Response.json({ received: true }, { headers: accountHeaders() });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/api/dashboard') {
      try {
        return Response.json(await dashboardForRequest(request, url, env), { headers: dashboardHeaders() });
      } catch (error) {
        const status = Number.isInteger(error.status) ? error.status : 503;
        return Response.json({ error: error.message || 'Live forecast unavailable.' }, { status, headers: { 'cache-control': 'no-store' } });
      }
    }
    if (url.pathname === '/api/account' && request.method === 'GET') {
      try { return Response.json(publicAccount(await accountForRequest(request, env), env), { headers: accountHeaders() }); } catch (error) { return jsonError(error.message || 'Account service is unavailable.', 503); }
    }
    if (url.pathname === '/api/access/claim' && request.method === 'POST') return claimDailyPick(request, env);
    if (url.pathname === '/api/billing/checkout' && request.method === 'POST') return createCheckoutSession(request, url, env);
    if (url.pathname === '/api/billing/portal' && request.method === 'POST') return createBillingPortal(request, url, env);
    if (url.pathname === '/api/billing/webhook' && request.method === 'POST') return processStripeWebhook(request, env);
    const path = url.pathname;
    // The browser uses this non-sensitive, packaged reference only if both
    // live forecast paths are unavailable. Serving it prevents a total blank
    // board while the UI clearly labels it as non-live fallback data.
    if (path === '/dashboard-snapshot.json') {
      const snapshotAsset = assets[path];
      return new Response(request.method === 'HEAD' ? null : decode(snapshotAsset.body), { headers: { 'content-type': snapshotAsset.type, 'cache-control': 'no-store' } });
    }
    const asset = assets[path === '/' ? '/index.html' : path];
    if (!asset) return new Response('Not found', { status: 404 });
    return new Response(request.method === 'HEAD' ? null : decode(asset.body), {
      headers: { 'content-type': asset.type, 'cache-control': path === '/' ? 'no-cache' : 'public, max-age=3600' },
    });
  },
};
`
await mkdir(new URL('../dist/server/', import.meta.url), { recursive: true })
await writeFile(new URL('../dist/server/index.js', import.meta.url), worker)
