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
const CACHE_TTL_MS = 4 * 60 * 60 * 1000;
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

async function fetchHourlyForecasts(stations) {
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
  const response = await fetch(endpoint, { headers: { Accept: 'application/json' } });
  if (!response.ok) return null;
  const payload = await response.json();
  return Array.isArray(payload) ? payload.map((item) => item.hourly ?? {}) : [payload.hourly ?? {}];
}

function forecastForStation(station, targetDate, hourly) {
  const features = buildFeatures(hourly, targetDate);
  const baseline = Math.round(features?.['ncep_nbm_conus__tmax_f']);
  if (!Number.isFinite(baseline)) throw new Error('No NBM daily high available');
  const isCalibrated = station.stationId in modelArtifact.calibrationOffsetByStation;
  const rawHigh = Math.round(isCalibrated ? baseline + modelResidual(features, station.stationId) : baseline);
  const halfWidth = Math.max(2, Math.round(modelArtifact.conformalHalfwidthByStation[station.stationId] ?? 4));
  return makeForecast(station, targetDate, baseline, rawHigh, halfWidth, isCalibrated, isCalibrated ? 'Live NCEP NBM, HRRR, and GFS guidance with the validated station residual model.' : 'Live NCEP NBM baseline shown; station-specific calibration is not available.');
}

function makeForecast(station, targetDate, baseline, rawHigh, halfWidth, isCalibrated, reason) {
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
    uncertainty: halfWidth <= 2 ? 'Low' : 'Moderate',
    isCalibrated,
    currentObservedTemperatureF: null,
    observedHighSoFarF: null,
    observedLowSoFarF: null,
    intradayObservations: [],
    lastObservationAt: null,
    dataFreshness: 1,
    sourceAgreement: 1,
    reasonCodes: [reason],
    stabilityReason: 'four_hour_market_update_window',
  };
}

async function nwsForecastForStation(station, targetDate) {
  const grid = station.forecastGrid;
  const response = await fetch('https://api.weather.gov/gridpoints/' + grid.office + '/' + grid.x + ',' + grid.y + '/forecast', { headers: { Accept: 'application/geo+json', 'User-Agent': 'PolyWeather market forecast' } });
  if (!response.ok) throw new Error('NWS forecast unavailable');
  const periods = (await response.json()).properties?.periods ?? [];
  const candidates = periods.filter((period) => period.isDaytime && localDate(new Date(period.startTime), station.timezone) === targetDate).map((period) => Number(period.temperature)).filter(Number.isFinite);
  if (!candidates.length) throw new Error('No NWS daily high available');
  const baseline = Math.round(Math.max(...candidates));
  const adjustment = { KNYC: -2, KMDW: 1, KMIA: 1, KLAX: 4, KSFO: 2 }[station.stationId] ?? 0;
  const isCalibrated = station.stationId in modelArtifact.calibrationOffsetByStation;
  const halfWidth = Math.max(2, Math.round(modelArtifact.conformalHalfwidthByStation[station.stationId] ?? 4));
  return makeForecast(station, targetDate, baseline, baseline + adjustment, halfWidth, isCalibrated, isCalibrated ? 'Settlement-aware adjustment applied to the latest NOAA/NWS daily forecast.' : 'Latest NOAA/NWS daily forecast shown without station calibration.');
}

async function liveDashboard(targetDate) {
  const now = new Date();
  const today = localDate(now, 'America/New_York');
  const resolvedDate = /^\\d{4}-\\d{2}-\\d{2}$/.test(targetDate ?? '') ? targetDate : today;
  const hourlyForecasts = await fetchHourlyForecasts(snapshot.stationRegistry);
  const forecasts = hourlyForecasts
    ? snapshot.stationRegistry.flatMap((station, index) => { try { return [forecastForStation(station, resolvedDate, hourlyForecasts[index] ?? {})]; } catch { return []; } })
    : await mapWithConcurrency(snapshot.stationRegistry, (station) => nwsForecastForStation(station, resolvedDate));
  const usableForecasts = forecasts.filter((forecast) => !forecast?.error);
  if (!usableForecasts.length) throw new Error('Live market forecast data is temporarily unavailable.');
  return {
    ...snapshot,
    targetDate: resolvedDate,
    today,
    maxDate: addDays(today, 7),
    generatedAt: now.toISOString(),
    forecasts: usableForecasts,
    marketForecast: true,
    forecastInputs: hourlyForecasts ? 'Live NCEP NBM, HRRR, and GFS guidance with station-specific residual calibration where validated' : 'Latest NOAA/NWS daily guidance with settlement-aware adjustment where validated',
    validationTarget: 'Official NOAA/NCEI daily TMAX',
    releaseStatus: 'Market values are held in four-hour update windows to prevent refresh churn. Uncalibrated stations remain clearly marked as baseline-only.',
  };
}

async function dashboardResponse(url, targetDate) {
  const cacheKey = new Request(url.origin + '/api/dashboard-cache?date=' + encodeURIComponent(targetDate ?? 'today'));
  if (typeof caches !== 'undefined') {
    try {
      const cachedResponse = await caches.default.match(cacheKey);
      if (cachedResponse) return cachedResponse;
    } catch { /* Sites may disable the shared cache; the worker cache still prevents refresh churn. */ }
  }
  const memory = responseCache.get(targetDate ?? 'today');
  if (memory && Date.now() - memory.createdAt < CACHE_TTL_MS) return Response.json(await memory.value, { headers: { 'cache-control': 'public, max-age=14400, s-maxage=14400' } });
  const value = liveDashboard(targetDate);
  responseCache.set(targetDate ?? 'today', { createdAt: Date.now(), value });
  try {
    const dashboard = await value;
    const response = Response.json(dashboard, { headers: { 'cache-control': 'public, max-age=14400, s-maxage=14400' } });
    if (typeof caches !== 'undefined') {
      try { await caches.default.put(cacheKey, response.clone()); } catch { /* Use the four-hour worker cache when shared caching is unavailable. */ }
    }
    return response;
  } catch (error) {
    responseCache.delete(targetDate ?? 'today');
    throw error;
  }
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === '/api/dashboard') {
      try {
        return await dashboardResponse(url, url.searchParams.get('date'));
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
