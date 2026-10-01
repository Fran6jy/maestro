import { readFile } from "node:fs/promises";
import expectations from "@/data/live_expectations.json";

/** One strategy in the nightly snapshot (live/snapshot.py documents every field). */
export interface LiveStrategy {
  daily: { gross: number[]; ref: number[]; quoted: number[] };   // daily log returns
  trades: number;
  hit: number | null;
  bars_in_market: number;
  gross_pips: number;
  cost_ref_pips: number;
  cost_quoted_pips: number;
  position: number;
}

export interface LiveTrade {
  opened: string;
  closed: string;
  direction: number;
  gross_pips: number;
  cost_pips: number;
  gross_gbp: number;
  cost_gbp: number;
  open: boolean;
}

export interface LiveOrder {
  bar: string;
  units: number;
  status: string;
  vs_mid_pips: number | null;
  extra_pips: number | null;
  spread_pips: number | null;
  delay_s: number | null;
}

export interface Snapshot {
  generated_at: string;
  phase: "shakedown" | "trial";
  ref_cost_pips: number;
  start_gbp: number;
  code_commit: string | null;
  model: { deployed_at?: string; train_start?: string; train_end?: string; commit?: string };
  order_strategy: string | null;
  heartbeat: { at: string | null; error: string | null };
  start: string | null;
  last_bar?: string;
  days: string[];
  strategies: Record<string, LiveStrategy>;
  trades: Record<string, LiveTrade[]>;
  orders: {
    filled: number;
    failed: number;
    units_traded?: number;
    mean_vs_mid_pips?: number | null;
    mean_extra_pips?: number | null;
    total_extra_pips?: number | null;
    total_extra_gbp?: number | null;
    mean_spread_pips?: number | null;
    median_delay_s?: number | null;
    recent: LiveOrder[];
  };
  health: {
    bars_expected?: number;
    bars_recorded?: number;
    bars_missed?: number;
    bars_unscorable?: number;
  };
}

export interface Expectation {
  label: string;
  group: "maestro" | "msc" | "ref";
  desc: string;
  hit: number | null;
  trades_per_day: number;
  gross_pips_per_trade: number | null;
  ranges: number[][];        // [day-1][quantile] balance change in £
}

export interface Expectations {
  created_at: string;
  code_commit: string | null;
  backtest: { first_day: string; last_day: string; days: number };
  start_balance: number;
  quantiles: number[];
  horizon_days: number;
  strategies: Record<string, Expectation>;
}

export const EXPECT = expectations as unknown as Expectations;

/** Where the published snapshots live once the trial is public. */
export const LIVE_REPO = "https://github.com/Fran6jy/maestro-live";
export const SNAPSHOT_URL = "https://raw.githubusercontent.com/Fran6jy/maestro-live/main/snapshot.json";

/** The page is public only when LIVE_PUBLIC=1; before that it exists for local preview. */
export const LIVE_PUBLIC = process.env.LIVE_PUBLIC === "1";

/**
 * The latest snapshot: from the public repository (refreshed hourly), or for a local
 * preview from LIVE_SNAPSHOT_FILE. Null when neither is available.
 */
export async function getSnapshot(): Promise<Snapshot | null> {
  const file = process.env.LIVE_SNAPSHOT_FILE;
  if (file) {
    try {
      return JSON.parse(await readFile(file, "utf-8")) as Snapshot;
    } catch {
      return null;
    }
  }
  if (!LIVE_PUBLIC) return null;
  try {
    const res = await fetch(SNAPSHOT_URL, { next: { revalidate: 3600 } });
    return res.ok ? ((await res.json()) as Snapshot) : null;
  } catch {
    return null;
  }
}

/** Balance change in £ after each day, from daily log returns. */
export function balancePath(daily: number[], start: number): number[] {
  let acc = 0;
  return daily.map((r) => {
    acc += r;
    return start * (Math.exp(acc) - 1);
  });
}

/** Sum of daily log returns, as a balance change in £. */
export function balanceChange(daily: number[], start: number): number {
  return start * (Math.exp(daily.reduce((a, b) => a + b, 0)) - 1);
}

/** Where a value sits against the backtest's range for that many days. */
export function placement(value: number, q: number[]): "below" | "low" | "middle" | "high" | "above" {
  if (value < q[0]) return "below";
  if (value < q[1]) return "low";
  if (value <= q[3]) return "middle";
  if (value <= q[4]) return "high";
  return "above";
}
