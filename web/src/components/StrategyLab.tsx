"use client";

import { useMemo, useState } from "react";
import EquityChart, { type ChartMonth } from "./EquityChart";
import { RESULTS, STRATEGY, type StrategyKey } from "@/lib/data";
import { START, dailyNet, equity, finalBalance, monthlyNet, sharpe } from "@/lib/metrics";
import { gbp, int, pct, signed, toMs } from "@/lib/format";
import { useMoneyTween } from "@/lib/useTween";
import styles from "./StrategyLab.module.css";

const DATES = RESULTS.dates.map(toMs);
const MONTHS: ChartMonth[] = RESULTS.months.map((m) => ({
  start: toMs(m.start),
  end: toMs(m.end) + 86_400_000,
  label: m.label,
}));
const MAX_COST = 2;
const PRESETS = [
  { cost: 0, label: "No costs" },
  { cost: RESULTS.refCostPips, label: "Typical spread" },
  { cost: 1.5, label: "Spread + slippage" },
];
const HOLD = equity(dailyNet(STRATEGY.buy_hold, RESULTS.refCostPips));
const GROUPS = [
  { id: "maestro", label: "MAESTRO" },
  { id: "msc", label: "From my MSc" },
  { id: "ref", label: "Reference points" },
].filter((g) => RESULTS.strategies.some((st) => st.group === g.id));
const PERIOD = `${RESULTS.months[0].label} to ${RESULTS.months[RESULTS.months.length - 1].label}`;
const YEARS = Math.round(RESULTS.months.length / 12);
const FIRST: StrategyKey = (["risk_cost_filter", "maestro_top10", "logreg_lag5"] as StrategyKey[])
  .find((k) => STRATEGY[k]) ?? "logreg_lag5";

export default function StrategyLab() {
  const [key, setKey] = useState<StrategyKey>(FIRST);
  const [cost, setCost] = useState(RESULTS.refCostPips);
  const s = STRATEGY[key];
  const costLabel = `${cost.toFixed(2)} pips`;

  const series = useMemo(() => {
    const noDaily = dailyNet(s, 0);
    const atDaily = dailyNet(s, cost);
    return {
      no: equity(noDaily),
      at: equity(atDaily),
      sharpeAt: sharpe(atDaily),
      monthNet: monthlyNet(s, cost),
    };
  }, [s, cost]);

  const board = useMemo(
    () =>
      RESULTS.strategies
        .map((st) => ({ st, final: finalBalance(st, cost) }))
        .sort((a, b) => b.final - a.final),
    [cost],
  );

  const endAt = series.at[series.at.length - 1];
  const endNo = series.no[series.no.length - 1];
  const shown = useMoneyTween(endAt);
  const profitableMonths = series.monthNet.filter((v) => v > 0).length;
  const paidPips = s.trades * cost;
  const up = endAt >= START;

  return (
    <section id="lab" className="band" aria-labelledby="lab-title">
      <div className="shell">
        <div className="band-head">
          <p className="eyebrow">Strategy Lab</p>
          <h2 id="lab-title">Drag the cost. Watch the money.</h2>
          <p className="lede">
            Every strategy below traded EUR/USD for {RESULTS.months.length} months ({YEARS} years), each
            month without seeing it first.
            Pick one, then move the slider from <strong>no costs</strong> to what a real broker charges.
            That one change decides whether a strategy makes money.
          </p>
        </div>

        <aside className={styles.notice} role="note" aria-label="Correction">
          <p>
            <strong>Corrected on 1 October 2026.</strong> Two look-ahead bugs were found and fixed: macro
            data reached the models before it was published, and MAESTRO&rsquo;s market-regime labels used
            later prices. MAESTRO was re-run over all 20 years on the fixed code; the bugs barely changed
            its results. The other strategies use prices only and were never affected.
          </p>
        </aside>

        <div className={`panel panel-glow ${styles.lab}`}>
          <fieldset className={styles.picker}>
            <legend className="eyebrow">Strategy</legend>
            {GROUPS.map((group) => (
              <div key={group.id} className={styles.group}>
                <p className={styles.groupLabel}>{group.label}</p>
                {RESULTS.strategies
                  .filter((st) => st.group === group.id)
                  .map((st) => {
                    const f = board.find((b) => b.st.key === st.key)!.final;
                    return (
                      <label key={st.key} className={`${styles.option} ${st.key === key ? styles.active : ""}`}>
                        <input
                          type="radio"
                          name="strategy"
                          value={st.key}
                          checked={st.key === key}
                          onChange={() => setKey(st.key)}
                        />
                        <span className={styles.optName}>{st.label}</span>
                        <span className={`num ${styles.optVal} ${st.trades === 0 ? styles.idle : f >= START ? styles.gain : styles.loss}`}>
                          {st.trades === 0 ? "No trades" : gbp(f)}
                        </span>
                      </label>
                    );
                  })}
              </div>
            ))}
          </fieldset>

          <div className={styles.main}>
            <div className={styles.costRow}>
              <div className={styles.costHead}>
                <label htmlFor="cost" className="eyebrow">Cost per trade</label>
                <output htmlFor="cost" className={`num ${styles.costVal}`}>
                  {cost.toFixed(2)} <span>pips</span>
                </output>
              </div>
              <input
                id="cost"
                type="range"
                min={0}
                max={MAX_COST}
                step={0.05}
                value={cost}
                onChange={(e) => setCost(Number(e.target.value))}
                className={styles.slider}
                style={{ "--p": `${(cost / MAX_COST) * 100}%` } as React.CSSProperties}
                aria-valuetext={`${cost.toFixed(2)} pips per round trip`}
              />
              <div className={styles.presets} role="group" aria-label="Cost presets">
                {PRESETS.map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    className={styles.preset}
                    aria-pressed={Math.abs(cost - p.cost) < 1e-9}
                    onClick={() => setCost(p.cost)}
                  >
                    {p.label} <span className="num">{p.cost.toFixed(1)}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.headline} aria-live="polite">
              <p className={styles.headLead}>£10,000 traded with {s.label}, {PERIOD}</p>
              <p className={`num ${styles.big} ${s.trades === 0 ? styles.idle : up ? styles.gain : styles.cool}`}>{gbp(shown)}</p>
              <p className={styles.headSub}>
                at {cost.toFixed(2)} pips per trade · <span className={styles.amberText}>{gbp(endNo)}</span> with no costs
              </p>
            </div>

            <p className={styles.desc}>{s.desc}</p>

            <div className={styles.legend} aria-hidden="true">
              <span><i className={styles.swNo} />No costs</span>
              <span><i className={styles.swAt} />At {cost.toFixed(2)} pips</span>
              <span><i className={styles.swHold} />Buy and hold</span>
              <span><i className={styles.swGap} />Paid in costs</span>
            </div>

            <EquityChart
              isoDates={RESULTS.dates}
              dates={DATES}
              noCost={series.no}
              atCost={series.at}
              hold={key === "buy_hold" ? undefined : HOLD}
              months={MONTHS}
              monthNet={series.monthNet}
              costLabel={costLabel}
              title={`£10,000 traded with ${s.label}: ${gbp(endNo)} with no costs, ${gbp(endAt)} at ${costLabel} per trade.`}
            />
            <p className={styles.stripNote}>
              Each block under the chart is one test month: green made money at this cost, red lost it.
            </p>

            <dl className={styles.facts}>
              <div>
                <dt>Right about the next move</dt>
                <dd className="num">{pct(s.hit)}</dd>
              </div>
              <div>
                <dt>Trades</dt>
                <dd className="num">{int(s.trades)}</dd>
              </div>
              <div>
                <dt>Months in profit</dt>
                <dd className="num">
                  {profitableMonths}
                  <small> of {series.monthNet.length}</small>
                </dd>
              </div>
              <div>
                <dt>Sharpe ratio</dt>
                <dd className="num">{signed(series.sharpeAt)}</dd>
              </div>
              <div>
                <dt>Breaks even at</dt>
                <dd className="num">
                  {s.grossPerTrade > 0 ? `${s.grossPerTrade.toFixed(2)}` : "never"}
                  {s.grossPerTrade > 0 && <small> pips</small>}
                </dd>
              </div>
            </dl>
            <p className={styles.paid}>
              {key !== "buy_hold"
                ? `${int(s.trades)} trades × ${cost.toFixed(2)} pips = ${int(paidPips)} pips paid to the market.`
                : `Buy and hold trades once a month, so costs barely touch it.`}
            </p>
          </div>
        </div>

        <div className={styles.board}>
          <div className={styles.boardHead}>
            <h3>Every strategy at {cost.toFixed(2)} pips per trade</h3>
            <p>Final balance from £10,000. The dashed line is where you started.</p>
          </div>
          <ol className={styles.rows}>
            {board.map(({ st, final }) => {
              const w = (Math.log10(Math.max(1, Math.min(30_000, final))) / Math.log10(30_000)) * 100;
              const idle = st.trades === 0;
              const bar = idle ? styles.barIdle : final >= START ? styles.barGain : styles.barLoss;
              return (
                <li key={st.key}>
                  <button
                    type="button"
                    className={`${styles.row} ${st.key === key ? styles.rowActive : ""}`}
                    onClick={() => setKey(st.key)}
                    aria-pressed={st.key === key}
                  >
                    <span className={styles.rowName}>{st.label}</span>
                    <span className={styles.rowTrack}>
                      <span
                        className={`${styles.rowBar} ${bar}`}
                        style={{ width: `${w}%` }}
                      />
                      <span className={styles.rowStart} style={{ left: `${(Math.log10(START) / Math.log10(30_000)) * 100}%` }} />
                    </span>
                    <span
                      className={`num ${styles.rowVal} ${idle ? styles.idle : ""}`}
                      title={idle ? `Never traded, so the ${gbp(final)} was never at risk` : undefined}
                    >
                      {idle ? "No trades" : gbp(final)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}
