import { Filter } from 'lucide-react'

// The headline number here is deliberately how much of the board was
// rejected. A board showing three plays out of twenty is the system working
// as intended, and it has to read that way instead of looking like
// seventeen failures.
const ORDER = ['ELITE', 'STRONG', 'PLAYABLE', 'MARGINAL', 'PASS', 'DATA_INSUFFICIENT']
const LABELS = {
  ELITE: 'Elite',
  STRONG: 'Strong',
  PLAYABLE: 'Playable',
  MARGINAL: 'Marginal',
  PASS: 'Pass',
  DATA_INSUFFICIENT: 'No data',
}

export default function BetSummary({ summary, mode, onModeChange }) {
  if (!summary) return null
  const percent = (value) => (value == null ? '—' : `${Math.round(value * 100)}%`)
  return (
    <section className="bet-summary" aria-labelledby="bet-summary-title">
      <div className="bet-summary-head">
        <div>
          <p className="section-label">SELECTIVITY</p>
          <h2 id="bet-summary-title">
            {summary.evaluated} markets analysed · {summary.recommended} recommended
          </h2>
          <p>
            Most markets are not worth a bet. This board recommends {percent(summary.coverageRate)} of what it
            evaluated.
          </p>
        </div>
        <label className="mode-select">
          <Filter size={15} aria-hidden="true" />
          <span className="sr-only">Selectivity mode</span>
          <select value={mode} onChange={(event) => onModeChange?.(event.target.value)}>
            <option value="standard">Standard</option>
            <option value="conservative">Conservative</option>
            <option value="very_conservative">Very conservative</option>
          </select>
        </label>
      </div>
      <div className="bet-summary-counts">
        {ORDER.filter((tier) => summary.counts[tier]).map((tier) => (
          <div key={tier} className={`bet-count bet-count-${tier.toLowerCase()}`}>
            <strong>{summary.counts[tier]}</strong>
            <span>{LABELS[tier]}</span>
          </div>
        ))}
      </div>
      <dl className="bet-summary-stats">
        <div>
          <dt>Average recommended probability</dt>
          <dd>{percent(summary.averageRecommendedProbability)}</dd>
        </div>
        <div>
          <dt>Average rejected probability</dt>
          <dd>{percent(summary.averagePassProbability)}</dd>
        </div>
        <div>
          <dt>Coverage</dt>
          <dd>{percent(summary.coverageRate)}</dd>
        </div>
      </dl>
    </section>
  )
}
