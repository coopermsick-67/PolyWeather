import { AlertTriangle, Ban, CheckCircle2, HelpCircle, MinusCircle, ShieldCheck } from 'lucide-react'

// PASS is a result, not a failure, so it gets a real badge of its own rather
// than an absence of one. Only genuinely recommended tiers are allowed to use
// the affirmative colour -- a green badge on a marginal market is the exact
// false confidence this whole layer exists to remove.
const TIER_META = {
  ELITE: { icon: ShieldCheck, tone: 'elite', short: 'ELITE' },
  STRONG: { icon: CheckCircle2, tone: 'strong', short: 'STRONG' },
  PLAYABLE: { icon: MinusCircle, tone: 'playable', short: 'PLAYABLE' },
  MARGINAL: { icon: AlertTriangle, tone: 'marginal', short: 'MARGINAL' },
  PASS: { icon: Ban, tone: 'pass', short: 'PASS' },
  DATA_INSUFFICIENT: { icon: HelpCircle, tone: 'unknown', short: 'NO DATA' },
}

export function BetBadge({ decision, compact = false }) {
  if (!decision) return <span className="bet-badge bet-badge-unknown"><HelpCircle size={14} aria-hidden="true" /><span>Not evaluated</span></span>
  const meta = TIER_META[decision.tier] ?? TIER_META.DATA_INSUFFICIENT
  const Icon = meta.icon
  // A DATA_INSUFFICIENT market stopped before its inputs were trustworthy.
  // Printing a probability beside it would present exactly the number we
  // just said we could not stand behind.
  const probability = decision.bucket && decision.tier !== 'DATA_INSUFFICIENT'
    ? `${Math.round(decision.bucket.probability * 100)}%`
    : null
  return (
    <span className={`bet-badge bet-badge-${meta.tone}`} title={decision.label}>
      <Icon size={14} aria-hidden="true" />
      <span>{compact ? meta.short : decision.label}</span>
      {probability && <b>{probability}</b>}
    </span>
  )
}

function ScoreBar({ component }) {
  const share = component.maxPoints > 0 ? component.points / component.maxPoints : 0
  return (
    <div className="score-row">
      <span>{component.name}</span>
      <i className="score-track" aria-hidden="true"><b style={{ width: `${Math.round(share * 100)}%` }} /></i>
      <b>{component.points.toFixed(1)}<small>/{component.maxPoints.toFixed(0)}</small></b>
    </div>
  )
}

export default function BetVerdict({ decision }) {
  if (!decision) {
    return (
      <div className="bet-verdict bet-verdict-unknown">
        <h3>Bet decision</h3>
        <p>The decision layer is disabled for this board, so no recommendation has been made.</p>
      </div>
    )
  }
  const meta = TIER_META[decision.tier] ?? TIER_META.DATA_INSUFFICIENT
  const blocking = decision.reasons.filter((reason) => reason.severity === 'critical')
  const cautions = decision.reasons.filter((reason) => reason.severity === 'high' || reason.severity === 'medium')
  const supporting = decision.reasons.filter((reason) => reason.severity === 'positive')
  return (
    <div className={`bet-verdict bet-verdict-${meta.tone}`}>
      <div className="bet-verdict-head">
        <div>
          <p className="section-label">BET DECISION</p>
          <h3>{decision.label}</h3>
        </div>
        <BetBadge decision={decision} compact />
      </div>

      {decision.bucket && decision.tier !== 'DATA_INSUFFICIENT' && (
        <dl className="bet-verdict-facts">
          <div><dt>Most likely range</dt><dd>{decision.bucket.label}°F</dd></div>
          <div><dt>Calibrated probability</dt><dd>{Math.round(decision.bucket.probability * 100)}%</dd></div>
          <div><dt>Edge distance</dt><dd>{decision.bucket.minimumEdgeDistanceF.toFixed(1)}°F</dd></div>
          <div><dt>Bet quality</dt><dd>{decision.qualityScore.toFixed(0)}/100</dd></div>
        </dl>
      )}

      {blocking.length > 0 && (
        <div className="bet-reasons bet-reasons-blocking">
          <h4>Why this is not a bet</h4>
          <ul>{blocking.map((reason) => <li key={reason.code}>{reason.message}</li>)}</ul>
        </div>
      )}
      {cautions.length > 0 && (
        <div className="bet-reasons bet-reasons-caution">
          <h4>Cautions</h4>
          <ul>{cautions.map((reason) => <li key={reason.code}>{reason.message}</li>)}</ul>
        </div>
      )}
      {supporting.length > 0 && (
        <div className="bet-reasons bet-reasons-supporting">
          <h4>Supporting evidence</h4>
          <ul>{supporting.map((reason) => <li key={reason.code}>{reason.message}</li>)}</ul>
        </div>
      )}

      {decision.components.length > 0 && (
        <details className="bet-breakdown">
          <summary>Score breakdown</summary>
          {decision.components.map((component) => <ScoreBar key={component.name} component={component} />)}
        </details>
      )}

      {decision.alternatives?.length > 1 && (
        <details className="bet-breakdown">
          <summary>Other ranges considered</summary>
          <div className="bet-alternatives">
            {decision.alternatives.map((item) => (
              <span key={item.label}>{item.label}°F <b>{Math.round(item.probability * 100)}%</b></span>
            ))}
          </div>
        </details>
      )}

      <p className="bet-verdict-note">
        Research only. A calibrated probability is a measured historical frequency, not a promise about this day.
      </p>
    </div>
  )
}
