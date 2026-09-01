export default function RangeBar({ low, high }) {
  return <div className="range" aria-label={`Four-degree planning range ${low} to ${high} degrees Fahrenheit`}><strong>{low}°</strong><div className="range-line" aria-hidden="true"><i /><b /><i /></div><strong>{high}°</strong></div>
}
