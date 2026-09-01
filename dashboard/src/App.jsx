import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ArrowRight, CalendarDays, CheckCircle2, ChevronDown, ChevronRight,
  CloudSun, Menu, Moon, RefreshCw, ShieldCheck, SunMedium, X,
} from 'lucide-react'
import { getDashboard } from './api'
import ForecastBrief from './components/ForecastBrief'
import ForecastControls from './components/ForecastControls'
import ForecastPreview from './components/ForecastPreview'
import ForecastTable from './components/ForecastTable'
import MarketCard from './components/MarketCard'
import IntradayChart from './components/IntradayChart'
import StationInsight from './components/StationInsight'
import AccuracyPanel from './components/AccuracyPanel'
import StatusPanel from './components/StatusPanel'
import TrendChart from './components/TrendChart'

function formatStationList(registry, limit = 8) {
  if (!registry?.length) return null
  const names = registry.map((station) => `${station.display_name} (${station.stationId})`)
  const shown = names.slice(0, limit)
  const last = shown.pop()
  const list = shown.length ? `${shown.join(', ')}, and ${last}` : last
  const remaining = registry.length - limit
  return remaining > 0 ? `${list}, plus ${remaining} more` : list
}

function formatIso(date) {
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day}`
}

function isoToday() {
  const now = new Date()
  return formatIso(new Date(now.getFullYear(), now.getMonth(), now.getDate()))
}

function addDays(iso, days) {
  const date = new Date(`${iso}T12:00:00`)
  date.setDate(date.getDate() + days)
  return formatIso(date)
}

function longDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })
}

function shortDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function dateChoice(iso, today) {
  const offset = Math.round((new Date(`${iso}T12:00:00`) - new Date(`${today}T12:00:00`)) / 86400000)
  if (offset === 0) return 'Today'
  if (offset === 1) return 'Tomorrow'
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' })
}

function useRoute() {
  const [route, setRoute] = useState(() => window.location.hash === '#/forecast' ? 'forecast' : 'home')
  useEffect(() => {
    const change = () => setRoute(window.location.hash === '#/forecast' ? 'forecast' : 'home')
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  return route
}

function initialTheme() {
  try {
    const saved = window.localStorage.getItem('polyweather-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* Storage can be unavailable in private browsing contexts. */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function useTheme() {
  const [theme, setTheme] = useState(initialTheme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { window.localStorage.setItem('polyweather-theme', theme) } catch { /* Preference remains for this visit. */ }
  }, [theme])
  return [theme, () => setTheme((current) => current === 'dark' ? 'light' : 'dark')]
}

function Logo() {
  return <a className="brand" href="#/" aria-label="PolyWeather home"><CloudSun aria-hidden="true" /><span>PolyWeather</span></a>
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
        </nav>
        <div className="header-actions">
          {isForecast && <label className="date-control"><CalendarDays size={17} /><input aria-label="Forecast date" type="date" min={today} max={maxDate} value={selectedDate} onChange={(event) => onDateChange(event.target.value)} /><span>{longDate(selectedDate)}</span><ChevronDown size={16} /></label>}
          <button className="theme-toggle" type="button" onClick={onThemeToggle} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} aria-pressed={theme === 'dark'} title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>{theme === 'dark' ? <SunMedium size={18} /> : <Moon size={18} />}</button>
          {isForecast ? <button className="icon-button" type="button" onClick={onRefresh} disabled={loading} aria-label="Refresh forecast"><RefreshCw className={loading ? 'spin' : ''} size={18} /></button> : <a className="button button-small" href="#/forecast">Open dashboard <ArrowRight size={16} /></a>}
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
    ['Which locations are included?', stationList ? `PolyWeather follows ${stationList}.` : 'PolyWeather follows a configured registry of settlement stations across the U.S. Open the dashboard to see the current list.'],
    ['What does the 4° range mean?', 'It is a fixed ±2°F planning band around the displayed high. In the untouched historical candidate backtest, it contained the observed high 67% of the time—not a guarantee for today.'],
    ['How often do forecasts update?', 'The dashboard recomputes from available forecast guidance when you refresh. Small one-degree shifts are intentionally held to keep the display stable.'],
    ['Is this an official weather forecast?', 'No. PolyWeather is an independent model. It is best used alongside official National Weather Service guidance for weather-safety decisions.'],
  ]
  return (
    <main>
      <section className="home-hero" aria-labelledby="hero-title">
        <div className="page-width hero-grid">
          <div className="hero-copy">
            <h1 id="hero-title">Know the high.<br />Plan with confidence.</h1>
            <p>Experimental daily-high estimates for {stationCopy}—built for the one number that shapes your day.</p>
            <div className="hero-actions"><a className="button" href="#/forecast">View today’s forecast <ArrowRight size={18} /></a><a className="text-link" href="#/how-it-works">See how it works <ArrowRight size={16} /></a></div>
            <p className="hero-note"><span />Model ranges and performance monitoring are shown alongside every forecast.</p>
          </div>
          <ForecastPreview data={data} loading={loading} error={error} />
        </div>
      </section>

      <section className="home-section matter-section">
        <div className="page-width two-column">
          <div><p className="section-label">WHY DAILY HIGHS</p><h2>One clear view of the temperature that matters most.</h2></div>
          <div className="benefit-list">
            <div><SunMedium aria-hidden="true" /><div><h3>Plan your day</h3><p>Know what to expect before you get dressed, travel, or set your schedule.</p></div></div>
            <div><CalendarDays aria-hidden="true" /><div><h3>See a full week ahead</h3><p>Move from today through the next seven days without resetting anything.</p></div></div>
            <div><ShieldCheck aria-hidden="true" /><div><h3>Use a four-degree planning range</h3><p>Each forecast pairs a whole-degree prediction with a fixed ±2°F range and its historical coverage.</p></div></div>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="home-section how-section">
        <div className="page-width"><p className="section-label">HOW IT WORKS</p><h2>Forecasts with context, not false certainty.</h2>
          <div className="steps">
            <article><span>01</span><h3>Read available weather guidance</h3><p>The model begins with time-aware forecast guidance and station-specific features.</p></article>
            <article><span>02</span><h3>Calibrate the daily high</h3><p>It learns a location-aware correction to the underlying forecast rather than predicting from scratch.</p></article>
            <article><span>03</span><h3>Show the number and its range</h3><p>You get a whole-degree high, a practical ±2°F planning range, and the wider calibrated model range.</p></article>
          </div>
        </div>
      </section>

      <section id="accuracy" className="home-section method-section">
        <div className="page-width method-grid">
          <div><p className="section-label">MODEL TRANSPARENCY</p><h2>We monitor the forecast after it is made.</h2><p>Forecasts are stored before the outcome is known, then scored against official daily observations. That keeps the performance view honest.</p><a className="text-link" href="#/forecast">Explore performance <ArrowRight size={16} /></a></div>
          <div className="method-panel"><CheckCircle2 /><div><strong>Daily high, local date</strong><span>Labels are official daily observations—not a max of rounded reports.</span></div><div><strong>Stable by design</strong><span>Small input changes do not churn the number every refresh.</span></div><div><strong>Shadow monitoring</strong><span>Performance is shown as monitored evidence, not a blanket accuracy promise.</span></div></div>
        </div>
      </section>

      <section className="home-section faq-section">
        <div className="page-width narrow"><p className="section-label">FAQ</p><h2>Useful answers, before you open the dashboard.</h2>
          <div className="faq-list">{faqs.map(([question, answer], index) => <article key={question}><button type="button" aria-expanded={openFaq === index} onClick={() => setOpenFaq(openFaq === index ? null : index)}><span>{question}</span><ChevronDown /></button>{openFaq === index && <p>{answer}</p>}</article>)}</div>
        </div>
      </section>

      <section className="final-cta"><div className="page-width final-cta-inner"><div><CloudSun /><div><h2>Know the high. Plan with confidence.</h2><p>Open the daily forecast for {stationCount ? `all ${stationCount} locations` : 'every configured location'}.</p></div></div><a className="button" href="#/forecast">View today’s forecast <ArrowRight size={18} /></a></div></section>
    </main>
  )
}

function DayNavigator({ today, selectedDate, onSelect }) {
  return <nav className="day-navigator" aria-label="Choose a forecast day">{Array.from({ length: 8 }, (_, offset) => addDays(today, offset)).map((day) => <button key={day} type="button" className={day === selectedDate ? 'selected' : ''} onClick={() => onSelect(day)} aria-pressed={day === selectedDate}>{dateChoice(day, today)}<small>{shortDate(day)}</small></button>)}</nav>
}

function ForecastPage({ data, selectedDate, today, onSelectDate, loading, error, onRefresh, selectedStation, onSelectStation, stationFilter, onStationFilter, showBaseline, onShowBaseline, onCopy }) {
  const forecasts = data?.forecasts ?? []
  const registry = data?.stationRegistry ?? []
  const accuracy = data?.accuracy ?? []
  const calibratedIds = new Set(accuracy.map((station) => station.station))
  const visibleForecasts = stationFilter ? forecasts.filter((forecast) => forecast.station === stationFilter) : forecasts
  const selectedForecast = forecasts.find((forecast) => forecast.station === selectedStation)
  const selectedAccuracy = accuracy.find((station) => station.station === selectedForecast?.station)
  return <main className="forecast-page"><div className="page-width forecast-layout">
    <section className="forecast-main" aria-labelledby="forecast-title">
      <div className="forecast-heading"><div><p className="section-label">FORECAST WORKSPACE</p><h1 id="forecast-title">{longDate(selectedDate)}</h1><p>Settlement-station market analysis across {registry.length || 'all configured'} locations.</p></div><div className="refresh-inline"><button className="button button-quiet" type="button" onClick={onRefresh} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} />Refresh</button></div></div>
      <DayNavigator today={today} selectedDate={selectedDate} onSelect={onSelectDate} />
      <ForecastControls registry={registry} selectedStation={stationFilter} onSelectStation={onStationFilter} calibratedIds={calibratedIds} showBaseline={showBaseline} onShowBaseline={onShowBaseline} onCopy={onCopy} />
      <MarketCard registry={registry} selectedStation={selectedStation} forecast={selectedForecast} onSelectStation={onSelectStation} calibratedIds={calibratedIds} />
      <IntradayChart forecast={selectedForecast} />
      {error && <div className="alert" role="status"><strong>Live refresh could not finish.</strong><span>The last successful values remain visible. Try refreshing again in a moment.</span></div>}
      <ForecastTable forecasts={visibleForecasts} loading={loading} showBaseline={showBaseline} selectedStation={selectedStation} onSelectStation={onSelectStation} />
      <section className="performance-card" aria-labelledby="performance-title"><div className="card-title"><div><p className="section-label">MONITORED PERFORMANCE</p><h2 id="performance-title">Recent model performance</h2><p>Rolling mean absolute error compared with the raw NBM guidance.</p></div><a className="text-link" href="#/accuracy">About monitoring <ArrowRight size={15} /></a></div><TrendChart trend={data?.trend} loading={loading} /></section>
      <AccuracyPanel accuracy={accuracy} evidence={data?.modelEvidence} onSelectStation={onSelectStation} />
    </section>
    <aside className="forecast-side"><StationInsight forecast={selectedForecast} accuracy={selectedAccuracy} evidence={data?.modelEvidence} rankedCount={accuracy.length} /><ForecastBrief evidence={data?.modelEvidence} /><StatusPanel data={data} loading={loading} /></aside>
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

  const refresh = useCallback(async () => {
    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller
    setLoading(true)
    setError('')
    try {
      const next = await getDashboard(selectedDate, { signal: controller.signal })
      if (requestController.current !== controller) return
      setData(next)
      if (selectedDate < next.today) setSelectedDate(next.today)
    } catch (reason) {
      if (reason.name !== 'AbortError' && requestController.current === controller) setError(reason.message || 'Unable to refresh the forecast.')
    } finally {
      if (requestController.current === controller) setLoading(false)
    }
  }, [selectedDate])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => () => requestController.current?.abort(), [])
  useEffect(() => {
    if (!selectedStation && (data?.stationRegistry?.length || data?.forecasts?.length)) setSelectedStation(data.stationRegistry?.[0]?.stationId ?? data.forecasts[0].station)
  }, [data, selectedStation])
  useEffect(() => {
    const timer = window.setInterval(() => {
      const current = isoToday()
      setClockDate(current)
      setSelectedDate((value) => value < current ? current : value)
    }, 60_000)
    return () => window.clearInterval(timer)
  }, [])

  const copySummary = async () => {
    const day = longDate(selectedDate)
    const lines = (data?.forecasts ?? []).map((forecast) => `${forecast.station}: ${forecast.highF}°F · planning ${forecast.fourDegreeRangeLowF}°–${forecast.fourDegreeRangeHighF}°`).join('\n')
    try {
      await navigator.clipboard.writeText(`PolyWeather experimental forecast · ${day}\n${lines}`)
    } catch { setError('Could not copy the planning range. Select the forecast values directly instead.') }
  }
  return <div className="app"><a className="skip-link" href="#main-content">Skip to content</a><AppHeader route={route} selectedDate={selectedDate} today={today} maxDate={maxDate} onDateChange={setSelectedDate} onRefresh={refresh} loading={loading} theme={theme} onThemeToggle={toggleTheme} /><div id="main-content">{route === 'forecast' ? <ForecastPage data={data} selectedDate={selectedDate} today={today} onSelectDate={setSelectedDate} loading={loading} error={error} onRefresh={refresh} selectedStation={selectedStation} onSelectStation={setSelectedStation} stationFilter={stationFilter} onStationFilter={setStationFilter} showBaseline={showBaseline} onShowBaseline={setShowBaseline} onCopy={copySummary} /> : <HomePage data={data} loading={loading} error={error} />}</div><footer className="site-footer"><div className="page-width"><Logo /><div><a href="#/how-it-works">How it works</a><a href="#/accuracy">Model notes</a><a href="#/forecast">Dashboard</a></div><p>Independent forecast workspace. Use official guidance for weather-safety decisions.</p></div></footer></div>
}
