import { RESULTS, type Strategy } from "./data";

/** Starting balance used throughout the site. */
export const START = 10_000;

/**
 * Daily net log returns at any round-trip cost.
 * Costs are linear in the cost per trade, so any cost is recovered exactly from
 * the exported gross series and the series at the reference cost (0.8 pips).
 */
export function dailyNet(s: Strategy, costPips: number): Float64Array {
  const { gross, ref } = s.daily;
  const k = costPips / RESULTS.refCostPips;
  const out = new Float64Array(gross.length);
  for (let i = 0; i < gross.length; i++) {
    out[i] = (gross[i] - k * (gross[i] - ref[i])) / RESULTS.scale;
  }
  return out;
}

/** Growth of START given daily log returns. */
export function equity(daily: Float64Array, start = START): Float64Array {
  const out = new Float64Array(daily.length);
  let acc = 0;
  for (let i = 0; i < daily.length; i++) {
    acc += daily[i];
    out[i] = start * Math.exp(acc);
  }
  return out;
}

/** Annualised Sharpe ratio from daily returns. */
export function sharpe(daily: Float64Array): number {
  const n = daily.length;
  if (n < 2) return 0;
  let mean = 0;
  for (const v of daily) mean += v;
  mean /= n;
  let sq = 0;
  for (const v of daily) sq += (v - mean) ** 2;
  const sd = Math.sqrt(sq / (n - 1));
  return sd > 0 ? (mean / sd) * Math.sqrt(252) : 0;
}

/** Net pips per test month at a given cost. */
export function monthlyNet(s: Strategy, costPips: number): number[] {
  return s.monthly.grossPips.map((g, i) => g - s.monthly.trades[i] * costPips);
}

/** Final balance at a cost, without building the whole curve. */
export function finalBalance(s: Strategy, costPips: number): number {
  const d = dailyNet(s, costPips);
  let acc = 0;
  for (const v of d) acc += v;
  return START * Math.exp(acc);
}
