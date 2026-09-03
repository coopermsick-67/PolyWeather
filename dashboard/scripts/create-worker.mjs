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
  'xgb-worker-model.json',
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
const worker = `const assets = ${JSON.stringify(assets)};
const decode = (value) => Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
const snapshot = JSON.parse(new TextDecoder().decode(decode(assets['/dashboard-snapshot.json'].body)));
const modelArtifact = JSON.parse(new TextDecoder().decode(decode(assets['/xgb-worker-model.json'].body)));
const responseCache = new Map();
const CACHE_TTL_MS = 15 * 60 * 1000;
// A complete 20-station, three-model guidance batch can legitimately take
// longer than a single-point query. Keep the request bounded without forcing
// the whole dashboard into the lower-information NWS fallback too eagerly.
const UPSTREAM_TIMEOUT_MS = 12_000;
const MODELS = ['ncep_nbm_conus', 'ncep_hrrr_conus', 'ncep_gfs_seamless'];
const WEATHER_VARIABLES = ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'shortwave_radiation', 'surface_pressure'];

function localDate(value, timeZone) {
  const parts = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: '2-digit', day: '2-digit', timeZone }).formatToParts(value);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return values.year + '-' + values.month + '-' + values.day;
}

function addDays(iso, days) {
  const date = new Date(iso + 'T12:00:00Z');
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

async function fetchWithTimeout(input, init = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function finite(values) { return values.filter(Number.isFinite); }
function mean(values) { const valid = finite(values); return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : NaN; }
function max(values) { const valid = finite(values); return valid.length ? Math.max(...valid) : NaN; }
function min(values) { const valid = finite(values); return valid.length ? Math.min(...valid) : NaN; }
function sampleStd(values) { const valid = finite(values); if (valid.length < 2) return NaN; const average = mean(valid); return Math.sqrt(valid.reduce((sum, value) => sum + (value - average) ** 2, 0) / (valid.length - 1)); }

function dayOfYear(iso) {
  const [year, month, day] = iso.split('-').map(Number);
  return Math.round((Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 0)) / 86400000);
}

function buildFeatures(hourly, targetDate) {
  const dayIndexes = (hourly.time ?? []).flatMap((time, index) => time.startsWith(targetDate) ? [index] : []);
  if (!dayIndexes.length) return null;
  const features = {};
  for (const model of MODELS) {
    const temperature = dayIndexes.map((index) => hourly['temperature_2m_' + model]?.[index]);
    features[model + '__tmax_f'] = max(temperature);
    features[model + '__tmin_f'] = min(temperature);
    features[model + '__temp_mean_f'] = mean(temperature);
    for (const hour of [0, 3, 6, 9, 12, 15, 18, 21]) {
      const index = dayIndexes.find((candidate) => Number(hourly.time[candidate].slice(11, 13)) === hour);
      features[model + '__temp_' + String(hour).padStart(2, '0') + '_f'] = index === undefined ? NaN : hourly['temperature_2m_' + model]?.[index];
    }
    for (const variable of WEATHER_VARIABLES) {
      const values = dayIndexes.map((index) => hourly[variable + '_' + model]?.[index]);
      const prefix = model + '__' + variable;
      if (variable === 'wind_direction_10m') {
        features[prefix + '_sin_mean'] = mean(values.map((value) => Number.isFinite(value) ? Math.sin(value * Math.PI / 180) : NaN));
        features[prefix + '_cos_mean'] = mean(values.map((value) => Number.isFinite(value) ? Math.cos(value * Math.PI / 180) : NaN));
      } else if (variable === 'precipitation') {
        features[prefix + '_sum'] = finite(values).reduce((sum, value) => sum + value, 0);
        features[prefix + '_max'] = max(values);
      } else {
        features[prefix + '_mean'] = mean(values);
        features[prefix + '_max'] = max(values);
        features[prefix + '_min'] = min(values);
      }
    }
    features[model + '__availability'] = finite(temperature).length / dayIndexes.length;
    const morning = [0, 3, 6, 9].map((hour) => features[model + '__temp_' + String(hour).padStart(2, '0') + '_f']);
    const afternoon = [12, 15, 18, 21].map((hour) => features[model + '__temp_' + String(hour).padStart(2, '0') + '_f']);
    features[model + '__morning_mean_f'] = mean(morning);
    features[model + '__afternoon_mean_f'] = mean(afternoon);
    features[model + '__warming_f'] = features[model + '__afternoon_mean_f'] - features[model + '__morning_mean_f'];
  }
  for (const suffix of ['tmax_f', 'tmin_f', 'temp_mean_f']) {
    const values = MODELS.map((model) => features[model + '__' + suffix]);
    features['model_agreement__' + suffix + '_spread'] = max(values) - min(values);
    features['model_agreement__' + suffix + '_std'] = sampleStd(values);
    features['model_agreement__nbm_minus_hrrr__' + suffix] = features['ncep_nbm_conus__' + suffix] - features['ncep_hrrr_conus__' + suffix];
    features['model_agreement__nbm_minus_gfs__' + suffix] = features['ncep_nbm_conus__' + suffix] - features['ncep_gfs_seamless__' + suffix];
  }
  const day = dayOfYear(targetDate);
  features.dayofyear_sin = Math.sin(2 * Math.PI * day / 365.25);
  features.dayofyear_cos = Math.cos(2 * Math.PI * day / 365.25);
  features.month = Number(targetDate.slice(5, 7));
  return features;
}

function modelResidual(features, stationId) {
  const numeric = modelArtifact.numericColumns.map((column, index) => Number.isFinite(features[column]) ? features[column] : modelArtifact.numericMedians[index]);
  const indicators = modelArtifact.missingIndicatorIndices.map((index) => Number.isFinite(features[modelArtifact.numericColumns[index]]) ? 0 : 1);
  const stationVector = modelArtifact.stationCategories.map((station) => station === stationId ? 1 : 0);
  const values = [...numeric, ...indicators, ...stationVector];
  const booster = modelArtifact.booster.learner.gradient_booster.model;
  let prediction = Number(String(modelArtifact.booster.learner.learner_model_param.base_score).replace(/[^0-9eE+.-]/g, ''));
  for (const tree of booster.trees) {
    let node = 0;
    while (tree.left_children[node] !== -1) {
      const value = values[tree.split_indices[node]];
      node = !Number.isFinite(value) || value < tree.split_conditions[node] ? tree.left_children[node] : tree.right_children[node];
    }
    prediction += tree.split_conditions[node];
  }
  return prediction + (modelArtifact.calibrationOffsetByStation[stationId] ?? modelArtifact.calibrationOffset);
}

function sourceEvidence(features, sourceName) {
  const highs = MODELS.map((model) => features?.[model + '__tmax_f']).filter(Number.isFinite);
  const spread = highs.length > 1 ? Math.max(...highs) - Math.min(...highs) : null;
  return {
    sourceName,
    sourceCount: highs.length,
    modelSpreadF: spread,
    sourceAgreement: spread === null ? null : Math.max(0, Math.min(1, 1 - spread / 12)),
    dataFreshness: 1,
    hasCompleteCoreGuidance: MODELS.every((model) => Number.isFinite(features?.[model + '__tmax_f']) && features?.[model + '__availability'] >= .95),
  };
}

function consensusHigh(features) {
  const nbm = features?.ncep_nbm_conus__tmax_f;
  const hrrr = features?.ncep_hrrr_conus__tmax_f;
  const gfs = features?.ncep_gfs_seamless__tmax_f;
  if (![nbm, hrrr, gfs].every(Number.isFinite)) return nbm;
  // Archive-tested 50/25/25 consensus: 2.378°F MAE versus 2.651°F raw NBM.
  return .5 * nbm + .25 * hrrr + .25 * gfs;
}

function adaptiveHalfWidth(baseHalfWidth, evidence, calibrated) {
  const spreadPenalty = Number.isFinite(evidence.modelSpreadF) ? Math.min(3, Math.ceil(evidence.modelSpreadF / 3)) : 3;
  const missingPenalty = evidence.sourceCount >= 3 ? 0 : 2;
  return Math.max(2, Math.min(8, Math.round(baseHalfWidth + spreadPenalty + missingPenalty + (calibrated ? 0 : 1))));
}

async function mapWithConcurrency(items, mapper, limit = 4) {
  const results = new Array(items.length).fill(null);
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      try { results[index] = await mapper(items[index]); } catch (error) { results[index] = { error: items[index].stationId + ': ' + (error.message || 'forecast unavailable') }; }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function hourlyEndpoint(stations) {
  const endpoint = new URL('https://api.open-meteo.com/v1/forecast');
  for (const [key, value] of Object.entries({
    latitude: stations.map((station) => station.latitude).join(','),
    longitude: stations.map((station) => station.longitude).join(','),
    hourly: WEATHER_VARIABLES.join(','),
    temperature_unit: 'fahrenheit',
    timezone: 'auto',
    forecast_days: '16',
    models: MODELS.join(','),
  })) endpoint.searchParams.set(key, String(value));
  return endpoint;
}

async function requestHourlyForecasts(stations) {
  const endpoint = hourlyEndpoint(stations);
  try {
    const response = await fetchWithTimeout(endpoint, { headers: { Accept: 'application/json', 'User-Agent': 'WeatherPicks forecast dashboard' } });
    if (!response.ok) return null;
    const payload = await response.json();
    return Array.isArray(payload) ? payload.map((item) => item.hourly ?? {}) : [payload.hourly ?? {}];
  } catch {
    return null;
  }
}

async function fetchHourlyForecasts(stations) {
  const batch = await requestHourlyForecasts(stations);
  if (batch?.length === stations.length) return batch;
  // Some edge networks reject a large multi-location query even though the
  // same provider accepts individual requests. Recover per station rather
  // than silently switching every city to a single-source NWS fallback.
  const individual = await mapWithConcurrency(stations, async (station) => {
    const result = await requestHourlyForecasts([station]);
    return result?.[0] ?? null;
  }, 10);
  return individual.some(Boolean) ? individual : null;
}

function forecastForStation(station, targetDate, hourly) {
  const features = buildFeatures(hourly, targetDate);
  const nbm = features?.['ncep_nbm_conus__tmax_f'];
  if (!Number.isFinite(nbm)) throw new Error('No NBM daily high available');
  const evidence = sourceEvidence(features, 'NCEP NBM, HRRR, and GFS via Open-Meteo');
  const baseline = Math.round(consensusHigh(features));
  const isCalibrated = station.stationId in modelArtifact.calibrationOffsetByStation && evidence.hasCompleteCoreGuidance;
  const rawHigh = Math.round(isCalibrated ? nbm + modelResidual(features, station.stationId) : baseline);
  const baseHalfWidth = Math.round(modelArtifact.conformalHalfwidthByStation[station.stationId] ?? 4);
  const halfWidth = adaptiveHalfWidth(baseHalfWidth, evidence, isCalibrated);
  return makeForecast(station, targetDate, baseline, rawHigh, halfWidth, isCalibrated, isCalibrated ? 'Archive-calibrated residual model applied to complete NBM, HRRR, and GFS guidance.' : evidence.sourceCount >= 3 ? 'Archive-tested NBM/HRRR/GFS weighted consensus shown; this station has no residual-model validation.' : 'Incomplete multi-model guidance; station-specific residual correction is deliberately disabled.', evidence);
}

function makeForecast(station, targetDate, baseline, rawHigh, halfWidth, isCalibrated, reason, evidence = {}) {
  const high = rawHigh;
  const prior = snapshot.forecasts.find((forecast) => forecast.station === station.stationId) ?? {};
  return {
    ...prior,
    station: station.stationId,
    city: station.name,
    marketLocation: station.display_name,
    targetDate,
    highF: high,
    rawModelHighF: rawHigh,
    baselineHighF: baseline,
    modelDeltaF: rawHigh - baseline,
    rangeLowF: high - halfWidth,
    rangeHighF: high + halfWidth,
    fourDegreeRangeLowF: high - 2,
    fourDegreeRangeHighF: high + 2,
    modelRange: [high - halfWidth, high + halfWidth],
    uncertainty: isCalibrated ? (halfWidth <= 2 ? 'Low' : 'Moderate') : 'High',
    isCalibrated,
    currentObservedTemperatureF: null,
    observedHighSoFarF: null,
    observedLowSoFarF: null,
    intradayObservations: [],
    lastObservationAt: null,
    dataFreshness: evidence.dataFreshness ?? null,
    sourceAgreement: evidence.sourceAgreement ?? null,
    sourceCount: evidence.sourceCount ?? 0,
    modelSpreadF: evidence.modelSpreadF ?? null,
    sourceName: evidence.sourceName ?? 'Forecast source unavailable',
    dataQualityStatus: isCalibrated ? 'PROVISIONAL / SHADOW' : 'NO BET / UNVALIDATED',
    reasonCodes: [reason],
    stabilityReason: 'fifteen_minute_market_update_window',
  };
}

async function nwsForecastForStation(station, targetDate) {
  const grid = station.forecastGrid;
  const response = await fetchWithTimeout('https://api.weather.gov/gridpoints/' + grid.office + '/' + grid.x + ',' + grid.y + '/forecast', { headers: { Accept: 'application/geo+json', 'User-Agent': 'WeatherPicks market forecast' } });
  if (!response.ok) throw new Error('NWS forecast unavailable');
  const periods = (await response.json()).properties?.periods ?? [];
  const candidates = periods.filter((period) => period.isDaytime && localDate(new Date(period.startTime), station.timezone) === targetDate).map((period) => Number(period.temperature)).filter(Number.isFinite);
  if (!candidates.length) throw new Error('No NWS daily high available');
  const baseline = Math.round(Math.max(...candidates));
  const evidence = { dataFreshness: 1, sourceName: 'NOAA/NWS daily forecast', sourceCount: 1, sourceAgreement: null, modelSpreadF: null };
  const halfWidth = adaptiveHalfWidth(Math.round(modelArtifact.conformalHalfwidthByStation[station.stationId] ?? 4), evidence, false);
  return makeForecast(station, targetDate, baseline, baseline, halfWidth, false, 'Latest NOAA/NWS daily forecast shown as a single-source fallback; no heuristic residual adjustment is applied.', evidence);
}

async function stationObservations(station, targetDate, now) {
  if (localDate(now, station.timezone) !== targetDate) return [];
  const start = addDays(targetDate, -1) + 'T00:00:00Z';
  const response = await fetchWithTimeout('https://api.weather.gov/stations/' + station.stationId + '/observations?start=' + encodeURIComponent(start) + '&limit=100', { headers: { Accept: 'application/geo+json', 'User-Agent': 'WeatherPicks market forecast' } });
  if (!response.ok) return [];
  return ((await response.json()).features ?? []).map((feature) => {
    const properties = feature.properties ?? {};
    const celsius = Number(properties.temperature?.value);
    return { time: properties.timestamp, temperatureF: Number.isFinite(celsius) ? celsius * 9 / 5 + 32 : NaN };
  }).filter((item) => item.time && Number.isFinite(item.temperatureF) && localDate(new Date(item.time), station.timezone) === targetDate).sort((a, b) => new Date(a.time) - new Date(b.time));
}

function applyObservations(forecast, observations) {
  if (!observations.length) return forecast;
  const temperatures = observations.map((item) => item.temperatureF);
  const observedHigh = Math.round(max(temperatures));
  const observedLow = Math.round(min(temperatures));
  const current = observations.at(-1);
  const high = Math.max(forecast.highF, observedHigh);
  const floorShift = high - forecast.highF;
  return {
    ...forecast,
    highF: high,
    rawModelHighF: high,
    rangeLowF: Math.max(observedHigh, forecast.rangeLowF + floorShift),
    rangeHighF: Math.max(observedHigh, forecast.rangeHighF + floorShift),
    fourDegreeRangeLowF: Math.max(observedHigh, forecast.fourDegreeRangeLowF + floorShift),
    fourDegreeRangeHighF: Math.max(observedHigh, forecast.fourDegreeRangeHighF + floorShift),
    currentObservedTemperatureF: Math.round(current.temperatureF * 10) / 10,
    observedHighSoFarF: observedHigh,
    observedLowSoFarF: observedLow,
    intradayObservations: observations.slice(-48),
    lastObservationAt: current.time,
    reasonCodes: [...forecast.reasonCodes, floorShift > 0 ? "Same-station NWS observation set a hard floor on today's high." : "Same-station NWS observations confirm the current forecast remains above the observed high."],
  };
}

function cachedForecastForStation(station, targetDate) {
  const prior = snapshot.forecasts.find((forecast) => forecast.station === station.stationId);
  const high = Number(prior?.highF);
  if (!Number.isFinite(high) || targetDate !== snapshot.targetDate) return null;
  const isCalibrated = station.stationId in modelArtifact.calibrationOffsetByStation;
  const halfWidth = Math.max(2, Math.round(modelArtifact.conformalHalfwidthByStation[station.stationId] ?? 4));
  return {
    ...makeForecast(station, targetDate, high, high, halfWidth, false, 'Live weather guidance is temporarily unavailable. The only same-date published snapshot is marked stale and should not be used as a new forecast.', { dataFreshness: 0, sourceName: 'Last same-date published snapshot', sourceCount: 0, sourceAgreement: null, modelSpreadF: null }),
    dataFreshness: 0,
    stabilityReason: 'last_known_market_snapshot',
  };
}

async function liveDashboard(targetDate) {
  const now = new Date();
  const today = localDate(now, 'America/New_York');
  const resolvedDate = /^\\d{4}-\\d{2}-\\d{2}$/.test(targetDate ?? '') ? targetDate : today;
  const hourlyForecasts = await fetchHourlyForecasts(snapshot.stationRegistry);
  const primaryForecasts = hourlyForecasts
    ? snapshot.stationRegistry.flatMap((station, index) => { try { return [forecastForStation(station, resolvedDate, hourlyForecasts[index] ?? {})]; } catch { return []; } })
    : [];
  const primaryByStation = new Map(primaryForecasts.map((forecast) => [forecast.station, forecast]));
  const missingStations = snapshot.stationRegistry.filter((station) => !primaryByStation.has(station.stationId));
  const nwsForecasts = missingStations.length
    ? await mapWithConcurrency(missingStations, (station) => nwsForecastForStation(station, resolvedDate), 10)
    : [];
  const liveForecasts = [...primaryForecasts, ...nwsForecasts.filter((forecast) => !forecast?.error)];
  const liveByStation = new Map(liveForecasts.map((forecast) => [forecast.station, forecast]));
  const usableForecasts = snapshot.stationRegistry.map((station) => liveByStation.get(station.stationId) ?? cachedForecastForStation(station, resolvedDate)).filter(Boolean);
  const observations = await mapWithConcurrency(snapshot.stationRegistry, (station) => stationObservations(station, resolvedDate, now), 8);
  const observationsByStation = new Map(snapshot.stationRegistry.map((station, index) => [station.stationId, Array.isArray(observations[index]) ? observations[index] : []]));
  const observedForecasts = usableForecasts.map((forecast) => applyObservations(forecast, observationsByStation.get(forecast.station) ?? []));
  if (!usableForecasts.length) throw new Error('Live market forecast data is temporarily unavailable.');
  return {
    ...snapshot,
    targetDate: resolvedDate,
    today,
    maxDate: addDays(today, 7),
    generatedAt: now.toISOString(),
    forecasts: observedForecasts,
    marketForecast: true,
    forecastInputs: primaryForecasts.length === snapshot.stationRegistry.length ? 'Live NCEP NBM, HRRR, and GFS guidance with an archive-tested consensus baseline, residual-model quality gates, adaptive uncertainty, and same-station NWS observations for today.' : liveForecasts.length ? 'Live guidance with station-level NOAA/NWS fallback, adaptive uncertainty, and same-station observations where available. A stale snapshot is used only when it matches the requested date.' : 'Live weather guidance is temporarily unavailable.',
    validationTarget: 'Official NOAA/NCEI daily TMAX',
    modelStatus: 'SHADOW_ONLY: residual MOS evaluated historically for KLAX, KMDW, KMIA, KNYC, and KSFO only.',
    releaseStatus: 'Forecasts revalidate every 15 minutes. Five stations have historical residual-model evaluation; the other configured stations use an archive-tested multi-model consensus when complete guidance is available. Same-station observed highs floor the current-day forecast, and range width expands with disagreement.',
  };
}

function responseHeaders() {
  return { 'cache-control': 'public, max-age=0, s-maxage=900, must-revalidate' };
}

async function dashboardResponse(url, targetDate, forceRefresh) {
  const cacheBucket = Math.floor(Date.now() / CACHE_TTL_MS);
  const cacheKey = new Request(url.origin + '/api/dashboard-cache?date=' + encodeURIComponent(targetDate ?? 'today') + '&bucket=' + cacheBucket);
  if (!forceRefresh && typeof caches !== 'undefined') {
    try {
      const cachedResponse = await caches.default.match(cacheKey);
      if (cachedResponse) return cachedResponse;
    } catch { /* Sites may disable the shared cache; the worker cache still prevents refresh churn. */ }
  }
  const memoryKey = (targetDate ?? 'today') + ':' + cacheBucket;
  const memory = responseCache.get(memoryKey);
  if (!forceRefresh && memory && Date.now() - memory.createdAt < CACHE_TTL_MS) return Response.json(await memory.value, { headers: responseHeaders() });
  const value = liveDashboard(targetDate);
  responseCache.set(memoryKey, { createdAt: Date.now(), value });
  try {
    const dashboard = await value;
    const response = Response.json(dashboard, { headers: responseHeaders() });
    if (typeof caches !== 'undefined') {
      try { await caches.default.put(cacheKey, response.clone()); } catch { /* Use the 15-minute worker cache when shared caching is unavailable. */ }
    }
    return response;
  } catch (error) {
    responseCache.delete(memoryKey);
    throw error;
  }
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === '/api/dashboard') {
      try {
        return await dashboardResponse(url, url.searchParams.get('date'), url.searchParams.get('refresh') === '1');
      } catch (error) {
        return Response.json({ error: error.message || 'Live forecast unavailable.' }, { status: 503, headers: { 'cache-control': 'no-store' } });
      }
    }
    const path = url.pathname;
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
