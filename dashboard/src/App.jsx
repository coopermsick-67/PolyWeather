import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ArrowRight, BarChart3, Bell, CalendarDays, CheckCircle2, ChevronDown, ChevronRight,
  CloudSun, Database, FileText, Menu, Moon, RefreshCw, Settings, ShieldCheck, SunMedium, X,
} from 'lucide-react'
import { getDashboard } from './api'
import { addDays, dateChoice, isoToday, longDate, shortDate } from './dateUtils'
import ForecastBrief from './components/ForecastBrief'
import ForecastControls from './components/ForecastControls'
import ForecastPreview from './components/ForecastPreview'
import ForecastTable from './components/ForecastTable'
import MarketCard from './components/MarketCard'
import IntradayChart from './components/IntradayChart'
import ForecastFanChart from './components/ForecastFanChart'
import StationInsight from './components/StationInsight'
import AccuracyPanel from './components/AccuracyPanel'
import StatusPanel from './components/StatusPanel'
import TrendChart from './components/TrendChart'
import BacktestWorkspace from './components/BacktestWorkspace'

function formatStationList(registry, limit = 8) {
  if (!registry?.length) return null
  const names = registry.map((station) => `${station.display_name} (${station.stationId})`)
  const shown = names.slice(0, limit)
  const last = shown.pop()
  const list = shown.length ? `${shown.join(', ')}, and ${last}` : last
  const remaining = registry.length - limit
  return remaining > 0 ? `${list}, plus ${remaining} more` : list
}

function useRoute() {
  const readRoute = () => ['forecast', 'backtest'].includes(window.location.hash.slice(2)) ? window.location.hash.slice(2) : 'home'
  const [route, setRoute] = useState(readRoute)
  useEffect(() => {
    const change = () => setRoute(readRoute())
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  return route
}

function initialTheme() {
  try {
    const saved = window.localStorage.getItem('weatherpicks-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* Storage can be unavailable in private browsing contexts. */ }
  return 'dark'
}

function useTheme() {
  const [theme, setTheme] = useState(initialTheme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { window.localStorage.setItem('weatherpicks-theme', theme) } catch { /* Preference remains for this visit. */ }
  }, [theme])
  return [theme, () => setTheme((current) => current === 'dark' ? 'light' : 'dark')]
}

function Logo() {
  return <a className="brand" href="#/" aria-label="WeatherPicks home"><CloudSun aria-hidden="true" /><span>WeatherPicks</span></a>
}

function WorkspaceNav({ route, theme, onThemeToggle }) {
  const navItems = [
    { href: '#/', label: 'Overview', icon: BarChart3, active: route === 'home' },
    { href: '#/forecast', label: 'Forecast board', icon: CloudSun, active: route === 'forecast' },
    { href: '#/backtest', label: 'Research backtest', icon: Database, active: route === 'backtest' },
    { href: '#/how-it-works', label: 'Methodology', icon: FileText, active: false },
    { href: '#/accuracy', label: 'Accuracy notes', icon: ShieldCheck, active: false },
  ]
  return <aside className="workspace-nav"><Logo /><p className="workspace-nav-label">Forecast research</p><nav aria-label="Workspace navigation">
    {navItems.map(({ href, label, icon: Icon, active }) => <a key={href} href={href} className={active ? 'active' : ''} aria-current={active ? 'page' : undefined}><Icon size={19} /><span>{label}</span></a>)}
  </nav><div className="workspace-nav-bottom"><button type="button" onClick={onThemeToggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}><Moon size={18} /><span>{theme === 'dark' ? 'Dark theme' : 'Light theme'}</span></button><span className="workspace-account"><small>Public forecast workspace</small></span></div></aside>
}

function AppHeader({ route, selectedDate, today, maxDate, onDateChange, onRefresh, loading, theme, onThemeToggle }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const close = () => setMenuOpen(false)
  const isForecast = route === 'forecast'

  return (
    <header className="site-header">
      <div className="header-inner">
        <Logo />
        <button className="mobile-menu" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X /> : <Menu />}</button>
        <nav className={menuOpen ? 'main-nav open' : 'main-nav'} aria-label="Primary navigation">
          <a href="#/" onClick={close}>Product</a>
          <a href="#/how-it-works" onClick={close}>How it works</a>
          <a href="#/accuracy" onClick={close}>Accuracy</a>
          <a className={isForecast ? 'active' : ''} href="#/forecast" onClick={close}>Dashboard</a>
          <a className={route === 'backtest' ? 'active' : ''} href="#/backtest" onClick={close}>Backtest</a>
        </nav>
        <div className="header-actions">
          {isForecast && <label className="date-control"><CalendarDays size={17} /><input aria-label="Forecast date" type="date" min={today} max={maxDate} value={selectedDate} onChange={(event) => onDateChange(event.target.value)} /><span>{longDate(selectedDate)}</span><ChevronDown size={16} /></label>}
          <button className="theme-toggle" type="button" onClick={onThemeToggle} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} aria-pressed={theme === 'dark'} title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>{theme === 'dark' ? <SunMedium size={18} /> : <Moon size={18} />}</button>
          {isForecast && <button className="icon-button" type="button" onClick={onRefresh} disabled={loading} aria-label="Refresh forecast"><RefreshCw className={loading ? 'spin' : ''} size={18} /></button>}
        </div>
      </div>
    </header>
  )
}

function HomePage({ data, loading, error }) {
  const [openFaq, setOpenFaq] = useState(null)
  useEffect(() => {
    const id = window.location.hash === '#/how-it-works' ? 'how-it-works' : window.location.hash === '#/accuracy' ? 'accuracy' : null
    if (id) window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }), 0)
  }, [])
  const registry = data?.stationRegistry
  const stationCount = registry?.length
  const stationCopy = stationCount ? `${stationCount} settlement stations` : 'settlement stations across the U.S.'
  const stationList = formatStationList(registry)
  const faqs = [
    ['Which locations are included?', stationList ? `WeatherPicks follows ${stationList}.` : 'WeatherPicks follows a configured registry of settlement stations across the U.S. Open the board to see the current list.'],
    ['What does the 4° range mean?', 'It is a fixed ±2°F planning band around the displayed high. In the untouched historical candidate backtest, it contained the observed high 67% of the time—not a guarantee for today.'],
    ['How often do forecasts update?', 'The dashboard recomputes from available forecast guidance when you refresh. Small one-degree shifts are intentionally held to keep the display stable.'],
    ['Are these guaranteed picks?', 'No. WeatherPicks is research only, not betting, financial, or weather-safety advice. Forecasts are experimental; verify settlement rules and official NWS guidance before acting.'],
  ]
  return (
    <main>
      <section className="home-hero" aria-labelledby="hero-title">
        <div className="page-width hero-grid">
          <div className="hero-copy">
            <h1 id="hero-title">Weather picks,<br />with the evidence in view.</h1>
            <p>Independent daily-high research for {stationCopy}. Compare the settlement station, forecast range, source coverage, and monitored history before considering a weather pick.</p>
            <div className="hero-actions"><a className="button" href="#/forecast">Explore today’s board <ArrowRight size={18} /></a><a className="text-link" href="#/backtest">Review the backtest <ArrowRight size={16} /></a></div>
            <p className="hero-note"><span />Information only—not betting, financial, or weather-safety advice. Outcomes are not guaranteed.</p>
          </div>
          <ForecastPreview data={data} loading={loading} error={error} />
        </div>
      </section>

      <section className="home-section matter-section">
        <div className="page-width two-column">
          <div><p className="section-label">THE PICK RESEARCH BOARD</p><h2>Build a weather pick from the evidence, not a single number.</h2></div>
          <div className="benefit-list">
            <div><SunMedium aria-hidden="true" /><div><h3>Know the settlement station</h3><p>Read the mapped location and local-date high before you compare any weather market rule.</p></div></div>
            <div><CalendarDays aria-hidden="true" /><div><h3>See the forecast context</h3><p>Compare the daily high, model spread, observations, and next seven days in one board.</p></div></div>
            <div><ShieldCheck aria-hidden="true" /><div><h3>Review the history</h3><p>Inspect the held-out archived-composite evaluation before treating any model signal as useful research.</p></div></div>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="home-section how-section">
        <div className="page-width"><p className="section-label">HOW TO REVIEW A PICK</p><h2>Evidence first. Certainty never.</h2>
          <div className="steps">
            <article><span>01</span><h3>Read available weather guidance</h3><p>The model begins with time-aware forecast guidance and station-specific features.</p></article>
            <article><span>02</span><h3>Calibrate the daily high</h3><p>It learns a location-aware correction to the underlying forecast rather than predicting from scratch.</p></article>
            <article><span>03</span><h3>Review the range and history</h3><p>Use the range, source quality, and backtest—then make your own decision. No result is guaranteed.</p></article>
          </div>
        </div>
      </section>

      <section id="accuracy" className="home-section method-section">
        <div className="page-width method-grid">
          <div><p className="section-label">MODEL TRANSPARENCY</p><h2>Every weather pick should face historical review.</h2><p>The archive pairs historical guidance composites with official daily observations. It is reviewable research, not a frozen record of the live, continuously refreshed forecast.</p><a className="text-link" href="#/backtest">Run the historical evaluation <ArrowRight size={16} /></a></div>
          <div className="method-panel"><CheckCircle2 /><div><strong>Daily high, local date</strong><span>Labels are official daily observations—not a max of rounded reports.</span></div><div><strong>Stable by design</strong><span>Small input changes do not churn the number every refresh.</span></div><div><strong>Shadow monitoring</strong><span>Performance is shown as monitored evidence, not a blanket accuracy promise.</span></div></div>
        </div>
      </section>

      <section className="home-section faq-section">
        <div className="page-width narrow"><p className="section-label">FAQ</p><h2>Useful answers, before you open the dashboard.</h2>
          <div className="faq-list">{faqs.map(([question, answer], index) => <article key={question}><button type="button" aria-expanded={openFaq === index} onClick={() => setOpenFaq(openFaq === index ? null : index)}><span>{question}</span><ChevronDown /></button>{openFaq === index && <p>{answer}</p>}</article>)}</div>
        </div>
      </section>

      <section className="final-cta"><div className="page-width final-cta-inner"><div><CloudSun /><div><h2>Review the weather evidence before the pick.</h2><p>Forecasts are experimental research—not a guarantee, recommendation, or safety advisory.</p></div></div><a className="button" href="#/forecast">Explore today’s board <ArrowRight size={18} /></a></div></section>
    </main>
  )
}

function DayNavigator({ today, maxDate, selectedDate, onSelect }) {
  const days = Array.from({ length: 8 }, (_, offset) => addDays(today, offset)).filter((day) => day <= maxDate)
  return <nav className="day-navigator" aria-label="Choose a forecast day">{days.map((day) => <button key={day} type="button" className={day === selectedDate ? 'selected' : ''} onClick={() => onSelect(day)} aria-pressed={day === selectedDate}>{dateChoice(day, today)}<small>{shortDate(day)}</small></button>)}</nav>
}

function signInWithChatGPT() {
  const returnTo = `${window.location.pathname}${window.location.search}${window.location.hash || '#/forecast'}`
  window.location.assign(`/signin-with-chatgpt?return_to=${encodeURIComponent(returnTo)}`)
}

function SubscriptionButton({ account, onSubscribe, busy }) {
  if (account?.role === 'admin') return <span className="plan-badge admin"><Crown size={15} />Admin access</span>
  if (!account?.billingConfigured) return <span className="billing-pending">Checkout is being configured.</span>
  return <button className="button" type="button" onClick={onSubscribe} disabled={busy}><LockKeyhole size={17} />Unlock all picks · $10/week</button>
}

function PickAccessPanel({ account, registry, selectedStation, selectedDate, onClaim, onSubscribe, busy, error }) {
  const station = registry.find((item) => item.stationId === selectedStation) ?? registry[0]
  if (!account?.authenticated) return <section className="access-panel"><div className="access-icon"><LockKeyhole /></div><div><p className="section-label">MEMBER ACCESS</p><h2>Sign in to start your free trial.</h2><p>Free members can unlock one settlement-station pick each day for seven days. Members get the complete WeatherPicks board for $10 per week.</p><button className="button" type="button" onClick={signInWithChatGPT}><LogIn size={17} />Sign in with ChatGPT</button></div></section>
  if (account.tier === 'free_expired') return <section className="access-panel"><div className="access-icon"><Crown /></div><div><p className="section-label">TRIAL COMPLETE</p><h2>Your seven-day free trial has ended.</h2><p>Subscribe for the full multi-city board, every forecast date, and ongoing access to weather-pick research.</p><SubscriptionButton account={account} onSubscribe={onSubscribe} busy={busy} /></div></section>
  return <section className="access-panel"><div className="access-icon"><CloudSun /></div><div><p className="section-label">YOUR FREE DAILY PICK</p><h2>Choose one station for {longDate(selectedDate)}.</h2><p>Your free trial includes one pick per calendar day for seven days. Once claimed, today’s station cannot be changed.</p>{station && <button className="button" type="button" onClick={() => onClaim(station.stationId, selectedDate)} disabled={busy}><CheckCircle2 size={17} />Unlock {station.display_name}</button>} <SubscriptionButton account={account} onSubscribe={onSubscribe} busy={busy} />{error && <p className="access-error" role="status">{error}</p>}</div></section>
}

function AccountPage({ account, onSubscribe, onPortal, busy, error }) {
  if (!account?.authenticated) return <main className="account-page"><section className="account-card"><p className="section-label">WEATHERPICKS MEMBERSHIP</p><h1>Your forecast access, in one place.</h1><p>Sign in to start a seven-day free trial. You get one settlement-station pick per day; full-board membership is $10 per week.</p><button className="button" type="button" onClick={signInWithChatGPT}><LogIn size={17} />Sign in with ChatGPT</button></section></main>
  const trialDate = account.trialEndsAt ? new Date(account.trialEndsAt).toLocaleDateString() : null
  return <main className="account-page"><section className="account-card"><p className="section-label">WEATHERPICKS MEMBERSHIP</p><div className="account-title"><div><h1>{account.role === 'admin' ? 'Administrator access' : account.tier === 'member' ? 'Full-board membership' : account.tier === 'free_trial' ? 'Free daily-pick trial' : 'Choose your plan'}</h1><p>{account.email}</p></div>{account.role === 'admin' ? <span className="plan-badge admin"><Crown size={15} />Admin</span> : <span className="plan-badge">{account.tier === 'member' ? 'Active member' : 'Free access'}</span>}</div>
    {account.role === 'admin' ? <p>You have full WeatherPicks access as the site administrator.</p> : account.tier === 'member' ? <p>Your subscription is confirmed as <strong>{account.subscriptionStatus}</strong>. You can view every available pick.</p> : account.tier === 'free_trial' ? <p>Your trial ends {trialDate}. Claim one settlement-station pick per day, or upgrade for the full board.</p> : <p>Your trial is complete. Upgrade to keep seeing the full board.</p>}
    <div className="account-actions">{account.role !== 'admin' && account.tier !== 'member' && <SubscriptionButton account={account} onSubscribe={onSubscribe} busy={busy} />}{account.role !== 'admin' && account.subscriptionStatus !== 'none' && <button className="button button-quiet" type="button" onClick={onPortal} disabled={busy}>Manage billing</button>}</div>{error && <p className="access-error" role="status">{error}</p>}</section><section className="account-details"><div><strong>Free trial</strong><span>1 pick per calendar day for 7 days</span></div><div><strong>Full board</strong><span>All available picks and forecast dates</span></div><div><strong>Membership price</strong><span>$10 per week, recurring until cancelled</span></div></section></main>
}

function ForecastPage({ data, selectedDate, today, maxDate, onSelectDate, loading, error, onRefresh, selectedStation, onSelectStation, stationFilter, onStationFilter, showBaseline, onShowBaseline, onCopy, onClaim, onSubscribe, billingBusy, billingError }) {
  const forecasts = data?.forecasts ?? []
  const registry = data?.stationRegistry ?? []
  const accuracy = data?.accuracy ?? []
  const calibratedIds = new Set(accuracy.map((station) => station.station))
  const visibleForecasts = stationFilter ? forecasts.filter((forecast) => forecast.station === stationFilter) : forecasts
  const selectedForecast = forecasts.find((forecast) => forecast.station === selectedStation)
  const selectedAccuracy = accuracy.find((station) => station.station === selectedForecast?.station)
  const locked = false
  return <main className="forecast-page"><div className="page-width forecast-layout">
    <section className="forecast-main" aria-labelledby="forecast-title">
      <div className="forecast-heading"><div><p className="section-label">FORECAST WORKSPACE</p><h1 id="forecast-title">{longDate(selectedDate)}</h1><p>Settlement-station market analysis across {registry.length || 'all configured'} locations.</p></div><div className="refresh-inline"><button className="button button-quiet" type="button" onClick={onRefresh} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} />Refresh</button></div></div>
      <DayNavigator today={today} maxDate={maxDate} selectedDate={selectedDate} onSelect={onSelectDate} />
      <ForecastControls registry={registry} selectedStation={stationFilter} onSelectStation={onStationFilter} calibratedIds={calibratedIds} showBaseline={showBaseline} onShowBaseline={onShowBaseline} onCopy={onCopy} />
      {!locked && <>
      <MarketCard registry={registry} selectedStation={selectedStation} forecast={selectedForecast} onSelectStation={onSelectStation} calibratedIds={calibratedIds} />
      <ForecastFanChart forecast={selectedForecast} />
      <IntradayChart forecast={selectedForecast} />
      {error && <div className="alert" role="status"><strong>Live refresh could not finish.</strong><span>The last successful values remain visible. Try refreshing again in a moment.</span></div>}
      <ForecastTable forecasts={visibleForecasts} loading={loading} showBaseline={showBaseline} selectedStation={selectedStation} onSelectStation={onSelectStation} />
      <section className="performance-card" aria-labelledby="performance-title"><div className="card-title"><div><p className="section-label">MONITORED PERFORMANCE</p><h2 id="performance-title">Recent model performance</h2><p>Rolling mean absolute error compared with the raw NBM guidance.</p></div><a className="text-link" href="#/accuracy">About monitoring <ArrowRight size={15} /></a></div><TrendChart trend={data?.trend} loading={loading} /></section>
      <AccuracyPanel accuracy={accuracy} evidence={data?.modelEvidence} onSelectStation={onSelectStation} selectedStation={selectedStation} />
      </>}
    </section>
    {!locked && <aside className="forecast-side"><StationInsight forecast={selectedForecast} accuracy={selectedAccuracy} evidence={data?.modelEvidence} rankedCount={accuracy.length} /><ForecastBrief evidence={data?.modelEvidence} /><StatusPanel data={data} loading={loading} /></aside>}
  </div></main>
}

export default function App() {
  const route = useRoute()
  const [theme, toggleTheme] = useTheme()
  const [selectedDate, setSelectedDate] = useState(isoToday)
  const [clockDate, setClockDate] = useState(isoToday)
  const [data, setData] = useState(null)
  const [selectedStation, setSelectedStation] = useState(null)
  const [stationFilter, setStationFilter] = useState(null)
  const [showBaseline, setShowBaseline] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const requestController = useRef(null)
  const today = data?.today && data.today >= clockDate ? data.today : clockDate
  const maxDate = data?.maxDate ?? addDays(today, 7)

  const refresh = useCallback(async (force = false) => {
    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller
    setLoading(true)
    setError('')
    try {
      const next = await getDashboard(selectedDate, { signal: controller.signal, force })
      if (requestController.current !== controller) return
      setData(next)
      if (selectedDate < next.today) setSelectedDate(next.today)
    } catch (reason) {
      if (reason.name !== 'AbortError' && requestController.current === controller) setError(reason.message || 'Unable to refresh the forecast.')
    } finally {
      if (requestController.current === controller) setLoading(false)
    }
  }, [selectedDate])

  useEffect(() => { void refresh(false) }, [refresh])
  useEffect(() => () => requestController.current?.abort(), [])
  useEffect(() => {
    if (!selectedStation && (data?.stationRegistry?.length || data?.forecasts?.length)) setSelectedStation(data.stationRegistry?.[0]?.stationId ?? data.forecasts[0].station)
  }, [data, selectedStation])
  useEffect(() => {
    const pick = data?.account?.dailyPick
    if (pick?.stationId) setSelectedStation(pick.stationId)
    if (pick?.targetDate) setSelectedDate(pick.targetDate)
  }, [data?.account?.dailyPick?.stationId, data?.account?.dailyPick?.targetDate])
  useEffect(() => {
    const timer = window.setInterval(() => {
      const current = isoToday()
      setClockDate(current)
      setSelectedDate((value) => value < current ? current : value)
    }, 60_000)
    return () => window.clearInterval(timer)
  }, [])
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh(false) }, 15 * 60 * 1000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const copySummary = async () => {
    const day = longDate(selectedDate)
    const lines = (data?.forecasts ?? []).map((forecast) => `${forecast.station}: ${forecast.highF}°F · planning ${forecast.fourDegreeRangeLowF}°–${forecast.fourDegreeRangeHighF}°`).join('\n')
    try {
      await navigator.clipboard.writeText(`WeatherPicks experimental research · ${day}\n${lines}`)
    } catch { setError('Could not copy the planning range. Select the forecast values directly instead.') }
  }
  const forceRefresh = () => { void refresh(true) }
  const claimPick = async (stationId, targetDate) => {
    setBillingBusy(true)
    setBillingError('')
    try {
      await claimDailyPick(stationId, targetDate)
      setSelectedStation(stationId)
      await refresh(true)
    } catch (reason) { setBillingError(reason.message || 'Could not unlock this daily pick.') } finally { setBillingBusy(false) }
  }
  const beginCheckout = async () => {
    setBillingBusy(true)
    setBillingError('')
    try {
      const session = await startCheckout()
      if (!session?.url) throw new Error('Checkout did not return a secure payment link.')
      window.location.assign(session.url)
    } catch (reason) { setBillingError(reason.message || 'Could not start checkout.') } finally { setBillingBusy(false) }
  }
  const manageBilling = async () => {
    setBillingBusy(true)
    setBillingError('')
    try {
      const session = await openBillingPortal()
      if (!session?.url) throw new Error('Billing portal did not return a secure link.')
      window.location.assign(session.url)
    } catch (reason) { setBillingError(reason.message || 'Could not open billing management.') } finally { setBillingBusy(false) }
  }
  return <div className="app app-shell"><a className="skip-link" href="#main-content">Skip to content</a><WorkspaceNav route={route} theme={theme} onThemeToggle={toggleTheme} /><div className="workspace-content"><AppHeader route={route} selectedDate={selectedDate} today={today} maxDate={maxDate} onDateChange={setSelectedDate} onRefresh={forceRefresh} loading={loading} theme={theme} onThemeToggle={toggleTheme} /><div id="main-content">{route === 'forecast' ? <ForecastPage data={data} selectedDate={selectedDate} today={today} maxDate={maxDate} onSelectDate={setSelectedDate} loading={loading} error={error} onRefresh={forceRefresh} selectedStation={selectedStation} onSelectStation={setSelectedStation} stationFilter={stationFilter} onStationFilter={setStationFilter} showBaseline={showBaseline} onShowBaseline={setShowBaseline} onCopy={copySummary} /> : route === 'backtest' ? <BacktestWorkspace registry={data?.stationRegistry ?? []} /> : <HomePage data={data} loading={loading} error={error} />}</div><footer className="site-footer"><div className="page-width"><Logo /><div><a href="#/how-it-works">How it works</a><a href="#/accuracy">Model notes</a><a href="#/forecast">Forecast board</a><a href="#/backtest">Backtest</a></div><p>Experimental research only—not betting, financial, or weather-safety advice.</p></div></footer></div></div>
}
