import results from "@/data/results.json";
import ridge from "@/data/ridge.json";
import msc from "@/data/msc.json";

export type StrategyKey =
  | "maestro_top10"
  | "maestro_ungated"
  | "maestro_gated"
  | "risk_cost_filter"
  | "logreg_lag5"
  | "bollinger_20_2"
  | "contrarian_3"
  | "linreg_lag5"
  | "linreg_lag1"
  | "sma_20_200"
  | "random"
  | "buy_hold";

export interface Strategy {
  key: StrategyKey;
  label: string;
  group: "maestro" | "msc" | "ref";
  desc: string;
  hit: number | null;       // null when a strategy never traded
  hitMsc: number | null;
  trades: number;
  exposure: number;
  grossPerTrade: number;
  daily: { gross: number[]; ref: number[] };
  monthly: { grossPips: number[]; trades: number[] };
}

export interface Month {
  start: string;
  end: string;
  label: string;
}

export interface Results {
  instrument: string;
  refCostPips: number;
  scale: number;
  dates: string[];
  months: Month[];
  strategies: Strategy[];
}

export interface Ridge {
  days: string[];
  paths: number[][];
}

export interface MscReplication {
  reported_hit: number;
  reported_hit_5lag: number;
  reported_sharpe: number;
  rerun: Record<"5dp" | "4dp", { hit: number; flat_share: number }>;
  sample: Record<"5dp" | "4dp", number[]>;
}

export const RESULTS = results as Results;
export const RIDGE = ridge as Ridge;
export const MSC = msc as MscReplication;

export const STRATEGY: Record<StrategyKey, Strategy> = Object.fromEntries(
  RESULTS.strategies.map((s) => [s.key, s]),
) as Record<StrategyKey, Strategy>;
