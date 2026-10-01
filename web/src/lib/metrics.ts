import { RESULTS, type Strategy } from "./data";

/** Starting balance used throughout the site. */
export const START = 10_000;

/**
 * Weekly net log returns at any round-trip cost.
 * Costs are linear in the cost per trade, so any cost is recovered exactly from
 * the exported gross series and the series at the reference cost (0.8 pips).
 */
export function weeklyNet(s: Strategy, costPips: number): Float64Array {
  const { gross, ref } = s.weekly;
  const k = costPips / RESULTS.refCostPips;
  const out = new Float64Array(gross.length);
  for (let i = 0; i < gross.length; i++) {
    out[i] = (gross[i] - k * (gross[i] - ref[i])) / RESULTS.scale;
  }
  return out;
}

/** Growth of START given log returns per period. */
export function equity(returns: Float64Array, start = START): Float64Array {
  const out = new Float64Array(returns.length);
  let acc = 0;
  for (let i = 0; i < returns.length; i++) {
    acc += returns[i];
    out[i] = start * Math.exp(acc);
  }
  return out;
}

/**
 * Annualised Sharpe ratio of daily returns at any cost, exactly, from the exported daily
 * sums: with r = g - k*d (g gross, d the cost at 0.8 pips, k = cost / 0.8),
 * mean = (Sg - k Sd) / n and var = (Sgg - 2k Sgd + k^2 Sdd - n mean^2) / (n - 1).
 */
export function sharpe(s: Strategy, costPips: number): number {
  const { n, g, d, gg, dd, gd } = s.moments;
  if (n < 2) return 0;
  const k = costPips / RESULTS.refCostPips;
  const mean = (g - k * d) / n;
  const variance = (gg - 2 * k * gd + k * k * dd - n * mean * mean) / (n - 1);
  return variance > 0 ? (mean / Math.sqrt(variance)) * Math.sqrt(252) : 0;
}

/** Net pips per test month at a given cost. */
export function monthlyNet(s: Strategy, costPips: number): number[] {
  return s.monthly.grossPips.map((g, i) => g - s.monthly.trades[i] * costPips);
}

/** Final balance at a cost, without building the whole curve. */
export function finalBalance(s: Strategy, costPips: number): number {
  const d = weeklyNet(s, costPips);
  let acc = 0;
  for (const v of d) acc += v;
  return START * Math.exp(acc);
}
