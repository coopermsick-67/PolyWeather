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

async function liveDashboard(targetDate, env) {
  const dashboard = await fetchFlaskDashboard(targetDate, env);
  const registry = Array.isArray(dashboard.stationRegistry) && dashboard.stationRegistry.length ? dashboard.stationRegistry : snapshot.stationRegistry;
  const forecastStations = new Set(dashboard.forecasts.map((forecast) => forecast.station));
  // dashboard_payload.py omits a station entirely (rather than fabricating a
  // value) when its live inputs are incomplete; surface which ones so the
  // frontend can say so instead of the station silently vanishing.
  const unavailableStations = registry.filter((station) => !forecastStations.has(station.stationId)).map((station) => station.stationId);
  return { ...dashboard, stationRegistry: registry, unavailableStations, marketForecast: true };
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
  const accountRecord = await accountForRequest(request, env);
  const account = publicAccount(accountRecord, env);
  if (!accountRecord) return dashboardSummary(url.searchParams.get('date'), account, null);
  const db = await requireDatabase(env);
  const accessDate = todayForAccess();
  const dailyPick = await existingDailyPick(db, accountRecord.id, accessDate);
  if (!account.canViewForecasts && !dailyPick) return dashboardSummary(url.searchParams.get('date'), account, null);
  const dashboard = await dashboardValue(url, url.searchParams.get('date'), url.searchParams.get('refresh') === '1', env);
  const allowedForecasts = account.canViewForecasts ? dashboard.forecasts : dashboard.forecasts.filter((forecast) => forecast.station === dailyPick.station_id && forecast.targetDate === dailyPick.target_date);
  return { ...dashboard, forecasts: allowedForecasts, account: { ...account, canViewForecasts: account.canViewForecasts || Boolean(dailyPick), dailyPick: dailyPick ? { stationId: dailyPick.station_id, targetDate: dailyPick.target_date, accessDate: dailyPick.access_date } : null } };
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
    if (path === '/dashboard-snapshot.json') return new Response('Not found', { status: 404 });
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
