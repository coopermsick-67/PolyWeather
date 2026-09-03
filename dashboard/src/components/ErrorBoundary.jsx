import { Component } from 'react'
import { AlertTriangle, RotateCw } from 'lucide-react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('WeatherPicks dashboard crashed:', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="app-crash" role="alert">
        <div className="app-crash-card">
          <AlertTriangle size={28} aria-hidden="true" />
          <h1>Something went wrong.</h1>
          <p>The dashboard hit an unexpected error and could not continue rendering. Your data was not affected.</p>
          <button type="button" className="button" onClick={() => window.location.reload()}>
            <RotateCw size={16} aria-hidden="true" /> Reload the dashboard
          </button>
        </div>
      </div>
    )
  }
}
