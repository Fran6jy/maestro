import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { MINUS, pct } from "@/lib/format";
import {
  EXPECT,
  LIVE_PUBLIC,
  LIVE_REPO,
  balanceChange,
  balancePath,
  getSnapshot,
  placement,
  type Snapshot,
} from "@/lib/live";
import styles from "./live.module.css";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Live trial",
  description:
    "Every strategy trading EUR/USD with £10,000 of pretend money at live prices, against what 20 years of history said would happen.",
  // Unlisted during the shakedown; LIVE_INDEX=1 lets search engines in once the trial proper starts.
  robots: { index: process.env.LIVE_INDEX === "1", follow: process.env.LIVE_INDEX === "1" },
  openGraph: {
    title: "MAESTRO live trial",
    description: "Every strategy trading EUR/USD with £10,000 of pretend money at live prices, against a forecast written before it started.",
    type: "website",
  },
};

const REPO = "https://github.com/Fran6jy/maestro";
// The version that places practice orders leads; MAESTRO's cost check and the two
// strongest MSc strategies sit beside it. Every strategy is in the full table.
const DEFAULT_MAESTRO = "maestro_top10";
const OTHERS = ["risk_cost_filter", "bollinger_20_2", "logreg_lag5"];

// ── Formatting ────────────────────────────────────────────────────────────────
function money(v: number, signed = true): string {
  const a = Math.abs(v);
  const s = a >= 100 ? Math.round(a).toLocaleString("en-GB") : a.toFixed(2);
  if (!signed) return `£${s}`;
  return `${v > 0.004 ? "+" : v < -0.004 ? MINUS : ""}£${s}`;
}

function pips(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  return `${v < 0 ? MINUS : ""}${Math.abs(v).toFixed(digits)}`;
}

function ukTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-GB", { timeZone: "Europe/London", hour: "2-digit", minute: "2-digit" });
}

function ukDay(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { timeZone: "Europe/London", weekday: "long", day: "numeric", month: "long" });
}

function longDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { timeZone: "UTC", day: "numeric", month: "long", year: "numeric" });
}

function label(key: string): string {
  return EXPECT.strategies[key]?.label ?? key;
}

const PLACEMENT_TEXT = {
  below: "Worse than 19 in 20 stretches of history",
  low: "Below the middle of the usual range",
  middle: "Right where history said it would be",
  high: "Above the middle of the usual range",
  above: "Better than 19 in 20 stretches of history",
} as const;

// ── The called shot: backtest range against the live path ─────────────────────
function RangeChart({ snap, k, start, hero = false }: { snap: Snapshot; k: string; start: number; hero?: boolean }) {
  const exp = EXPECT.strategies[k];
  const live = balancePath(snap.strategies[k]?.daily.ref ?? [], start);
  if (!exp) return null;
  const n = live.length;
  // The axis grows with the trial, so the live line is always the widest thing in the chart.
  const span = Math.min(EXPECT.horizon_days, Math.max(10, n + 5));
  const ranges = exp.ranges.slice(0, span);
  const [W, H] = hero ? [600, 330] : [460, 220];
  const L = hero ? 86 : 76, R = 12, T = 14, B = 30;
  const lo = Math.min(0, ...ranges.map((q) => q[0]), ...live);
  const hi = Math.max(0, ...ranges.map((q) => q[4]), ...live);
  const pad = (hi - lo) * 0.08 || 1;
  const y = (v: number) => T + ((hi + pad - v) / (hi - lo + 2 * pad)) * (H - T - B);
  const x = (d: number) => L + (d / span) * (W - L - R);
  const band = (a: number, b: number) =>
    [`${x(0)},${y(0)}`, ...ranges.map((q, i) => `${x(i + 1)},${y(q[b])}`),
     ...ranges.map((q, i) => `${x(i + 1)},${y(q[a])}`).reverse()].join(" ");
  const median = [`${x(0)},${y(0)}`, ...ranges.map((q, i) => `${x(i + 1)},${y(q[2])}`)].join(" ");
  const path = [`${x(0)},${y(0)}`, ...live.map((v, i) => `${x(i + 1)},${y(v)}`)].join(" ");
  // Zero always; the extremes only where they don't collide with it.
  const ticks = [0, ...[lo, hi].filter((v) => Math.abs(y(v) - y(0)) >= 18)];
  const last = live[n - 1] ?? 0;
  const where = n ? placement(last, exp.ranges[Math.min(n, EXPECT.horizon_days) - 1]) : "middle";
  return (
    <figure className={`panel ${hero ? styles.heroCard : ""} ${styles.chartCard}`}>
      <figcaption>
        <span className={styles.chartName}>{hero ? `${exp.label}, against what history predicted` : exp.label}</span>
        <span className={`${styles.chartValue} ${Math.abs(last) < 0.005 ? styles.flat : last > 0 ? styles.gain : styles.loss}`}>{money(last)}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`${exp.label}: ${money(last)} after ${n} days. ${PLACEMENT_TEXT[where]}.`}>
        <g className={styles.bands}>
          <polygon points={band(0, 4)} className={styles.bandOuter} />
          <polygon points={band(1, 3)} className={styles.bandInner} />
          <polyline points={median} className={styles.median} />
        </g>
        <line x1={L} x2={W - R} y1={y(0)} y2={y(0)} className={styles.zero} />
        {ticks.map((v) => (
          <text key={v} x={L - 8} y={y(v) + 4} textAnchor="end" className={styles.tick}>{money(v)}</text>
        ))}
        <text x={L} y={H - 8} className={styles.tick}>day 1</text>
        <text x={W - R} y={H - 8} textAnchor="end" className={styles.tick}>day {span}</text>
        {n > 0 && <polyline points={path} pathLength={1} className={`${styles.livePath} ${styles.draw}`} />}
        {n > 0 && <circle cx={x(n)} cy={y(last)} r={4.5} className={styles.liveDot} />}
      </svg>
      <p className={styles.chartNote}>
        {n ? PLACEMENT_TEXT[where] : "No days traded yet"}
        {hero && ". Shaded: where 20 years of history said it would usually be by each day. Line: what is actually happening."}
      </p>
    </figure>
  );
}

/** The expert layer's toggle: a real control, with its label following its state. */
function NumbersToggle() {
  return (
    <summary>
      <span className={styles.showLabel}>Show the numbers</span>
      <span className={styles.hideLabel}>Hide the numbers</span>
    </summary>
  );
}

// ── Sections ───────────────────────────────────────────────────────────────────
function NotStarted() {
  return (
    <div className={`shell ${styles.page}`}>
      <header className={styles.head}>
        <h1>Starting soon</h1>
        <p className="lede">
          Every strategy will trade the euro against the dollar with £10,000 of pretend money at live
          prices, against a forecast of what should happen written down before it starts. This page
          updates every night once the trial is running.
        </p>
      </header>
    </div>
  );
}

export default async function LivePage() {
  if (!LIVE_PUBLIC && !process.env.LIVE_SNAPSHOT_FILE) notFound();
  const snap = await getSnapshot();
  if (!snap || !snap.start || !snap.days.length) return <NotStarted />;

  const start = snap.start_gbp ?? EXPECT.start_balance;
  const maestroKey = snap.order_strategy ?? DEFAULT_MAESTRO;
  const featured = [maestroKey, ...OTHERS.filter((k) => k !== maestroKey)].filter((k) => snap.strategies[k]).slice(0, 4);
  const all = Object.keys(snap.strategies).sort((a, b) =>
    balanceChange(snap.strategies[b].daily.ref, start) - balanceChange(snap.strategies[a].daily.ref, start));
  const n = snap.days.length;
  const dayIdx = Math.min(n, EXPECT.horizon_days) - 1;
  const m = snap.strategies[maestroKey];
  const storyKey = m && m.trades > 0 ? maestroKey : "logreg_lag5";
  const story = snap.strategies[storyKey];
  const trades = snap.trades[maestroKey] ?? [];
  const tradeDay = trades.length ? trades[trades.length - 1].closed : null;
  const dayTrades = tradeDay ? trades.filter((t) => ukDay(t.closed) === ukDay(tradeDay)) : [];
  const h = snap.health;
  const o = snap.orders;
  const commitUrl = (c: string | null | undefined) => (c ? `${REPO}/tree/${c}` : REPO);

  return (
    <div className={`shell ${styles.page}`}>
      <header className={styles.hero}>
        <div className={styles.heroText}>
          <div className={styles.headTop}>
                <span className={`pill ${snap.phase === "trial" ? "pill-done" : "pill-next"}`}>
              {snap.phase === "trial" ? "Running" : "Shakedown"}
            </span>
          </div>
          <h1>Day {n}: where the pretend money stands</h1>
          <p className="lede">
            On {longDate(snap.start)} each strategy got <strong>£10,000 of pretend money</strong>. Every five
            minutes, each one decides whether to bet on the euro rising or falling against the dollar, at
            real prices. Updated every night; last updated {longDate(snap.generated_at)}.
          </p>
          {snap.phase === "shakedown" && (
            <p className={`panel ${styles.notice}`}>
              This is a shakedown: the setup is being tested before the trial proper, which starts after
              a final test on data no model has seen. These numbers are not results yet.
            </p>
          )}
        </div>
        <div className={styles.heroChart}>
          <RangeChart snap={snap} k={maestroKey} start={start} hero />
        </div>
      </header>

      {/* 1. Balances */}
      <section className={styles.section} aria-labelledby="balances">
        <h2 id="balances" className="visually-hidden">Balances</h2>
        <div className={styles.cards}>
          {featured.map((k) => {
            const s = snap.strategies[k];
            const change = balanceChange(s.daily.ref, start);
            const where = placement(change, EXPECT.strategies[k]?.ranges[dayIdx] ?? [0, 0, 0, 0, 0]);
            return (
              <div key={k} className={`panel ${styles.card}`}>
                <p className={styles.cardLabel}>{label(k)}</p>
                <p className={styles.cardValue}>{money(start + change, false)}</p>
                <p className={`num ${Math.abs(change) < 0.005 ? styles.flat : change > 0 ? styles.gain : styles.loss}`}>{money(change)} so far</p>
                <p className={styles.cardMeta}>{s.trades.toLocaleString("en-GB")} {s.trades === 1 ? "trade" : "trades"} · {PLACEMENT_TEXT[where].toLowerCase()}</p>
              </div>
            );
          })}
        </div>
        <details className={styles.more}>
          <NumbersToggle />
          <div className={styles.moreBody}>
            <p>
              Every strategy, in pips (a pip is 0.0001 dollars per euro). Fees are charged at the
              backtest&rsquo;s {snap.ref_cost_pips.toFixed(1)} pips per round trip; the last column
              charges the spread OANDA actually quoted at each trade instead. Balances above use the
              backtest&rsquo;s fee, so they compare directly with the 20-year results.
            </p>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr><th>Strategy</th><th>Trades</th><th>Right calls</th><th>Before fees</th><th>Fees</th>
                    <th>After fees</th><th>After quoted fees</th><th>£ after fees</th></tr>
                </thead>
                <tbody>
                  {all.map((k) => {
                    const s = snap.strategies[k];
                    return (
                      <tr key={k}>
                        <td>{label(k)}</td>
                        <td className="num">{s.trades}</td>
                        <td className="num">{pct(s.hit)}</td>
                        <td className="num">{pips(s.gross_pips)}</td>
                        <td className="num">{pips(s.cost_ref_pips)}</td>
                        <td className="num">{pips(s.gross_pips - s.cost_ref_pips)}</td>
                        <td className="num">{pips(s.gross_pips - s.cost_quoted_pips)}</td>
                        <td className="num">{money(balanceChange(s.daily.ref, start))}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      </section>

      {/* 2. The called shot */}
      <section className={styles.section} aria-labelledby="called">
        <div className={styles.sectionHead}>
          <h2 id="called">Did history call it?</h2>
          <p className="lede">
            Before the trial started, 20 years of history were used to write down where each strategy
            would usually be after any number of days. The shaded area is that usual range, the line is
            what is actually happening.
          </p>
        </div>
        <div className={styles.charts}>
          {featured.map((k) => <RangeChart key={k} snap={snap} k={k} start={start} />)}
        </div>
        <p className={styles.legend}>
          <span><span className={styles.keyOuter} /> 9 in 10 stretches of history</span>
          <span><span className={styles.keyInner} /> the middle half</span>
          <span><span className={styles.keyLive} /> live</span>
        </p>
        <details className={styles.more}>
          <NumbersToggle />
          <div className={styles.moreBody}>
            <p>
              The ranges come from every run of {n} consecutive trading days in the walk-forward backtest
              ({EXPECT.backtest.days.toLocaleString("en-GB")} days, {longDate(EXPECT.backtest.first_day)} to{" "}
              {longDate(EXPECT.backtest.last_day)}), with fees at the reference cost. They were written
              on {longDate(EXPECT.created_at)} and committed to the{" "}
              <a href={`${REPO}/blob/main/web/src/data/live_expectations.json`} target="_blank" rel="noreferrer">public repository</a>{" "}
              before the trial began, so they cannot have been adjusted to fit. A trading day is a UTC
              calendar day with at least one bar, the same unit in both.
            </p>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr><th>Strategy, day {n}</th><th>5%</th><th>25%</th><th>Middle</th><th>75%</th><th>95%</th><th>Live</th></tr>
                </thead>
                <tbody>
                  {featured.map((k) => {
                    const q = EXPECT.strategies[k]?.ranges[dayIdx];
                    if (!q) return null;
                    return (
                      <tr key={k}>
                        <td>{label(k)}</td>
                        {q.map((v, i) => <td key={i} className="num">{money(v)}</td>)}
                        <td className="num">{money(balanceChange(snap.strategies[k].daily.ref, start))}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      </section>

      {/* 3. Right and still losing */}
      <section className={styles.section} aria-labelledby="right">
        <div className={styles.sectionHead}>
          <h2 id="right">Being right isn&rsquo;t enough</h2>
          {story && (
            <p className="lede">
              {label(storyKey)} has called <strong>{pct(story.hit, 0)}</strong> of its moves right
              {(story.hit ?? 0) >= 0.5
                ? ", more often than not"
                : ` so far, against ${pct(EXPECT.strategies[storyKey]?.hit ?? null, 0)} over 20 years`}. Before
              fees it is {balanceChange(story.daily.gross, start) >= 0 ? "up" : "down"}{" "}
              <strong>{money(Math.abs(balanceChange(story.daily.gross, start)), false)}</strong>; its{" "}
              {story.trades} trades cost{" "}
              <strong>{money(balanceChange(story.daily.gross, start) - balanceChange(story.daily.ref, start), false)}</strong>{" "}
              in fees. Every trade pays a small fee, and small edges are eaten by it.
            </p>
          )}
        </div>
        <div className={styles.splits}>
          {(() => {
            // One scale for every row, so a bigger loss looks bigger.
            const rows = featured.map((k) => {
              const s = snap.strategies[k];
              const won = balanceChange(s.daily.gross, start);
              return { k, s, won, fees: won - balanceChange(s.daily.ref, start) };
            });
            const scale = Math.max(...rows.map((r) => Math.abs(r.won) + r.fees), 1e-9) * 1.02;   // room for the gap
            return rows.map(({ k, s, won, fees }) => (
              <div key={k} className={styles.split}>
                <p className={styles.splitLabel}>{label(k)}</p>
                {s.trades === 0 ? (
                  <p className={styles.small}>No trades yet</p>
                ) : (
                  <div className={styles.splitCol}>
                    <div className={styles.splitBar} aria-hidden="true">
                      <span className={won >= 0 ? styles.won : styles.lost}
                        style={{ width: `${(Math.abs(won) / scale) * 100}%` }} />
                      <span className={styles.fees} style={{ width: `${(fees / scale) * 100}%` }} />
                    </div>
                    <p className={`num ${styles.splitText}`}>
                      <span className={won >= 0 ? styles.gain : styles.lostText}>
                        {won >= 0 ? "won" : "lost"} {money(Math.abs(won), false)}
                      </span>{" "}
                      before fees · <span className={styles.feeText}>{money(fees, false)} in fees</span>
                    </p>
                  </div>
                )}
              </div>
            ));
          })()}
        </div>
        <details className={styles.more}>
          <NumbersToggle />
          <div className={styles.moreBody}>
            <p>
              &ldquo;Right calls&rdquo; counts the 5-minute bars a strategy held a position through and the
              price moved, and asks whether it moved the way the position bet. Ties (no move) are left out,
              exactly as in the backtest.
            </p>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead><tr><th>Strategy</th><th>Right calls, live</th><th>Right calls, 20 years</th><th>Trades a day, 20 years</th><th>Pips a trade before fees, 20 years</th></tr></thead>
                <tbody>
                  {featured.map((k) => {
                    const e = EXPECT.strategies[k];
                    return (
                      <tr key={k}>
                        <td>{label(k)}</td>
                        <td className="num">{pct(snap.strategies[k].hit)}</td>
                        <td className="num">{pct(e?.hit ?? null)}</td>
                        <td className="num">{e ? e.trades_per_day.toFixed(1) : "n/a"}</td>
                        <td className="num">{pips(e?.gross_pips_per_trade, 2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      </section>

      {/* 4. What MAESTRO did */}
      <section className={styles.section} aria-labelledby="did">
        <div className={styles.sectionHead}>
          <h2 id="did">What {label(maestroKey)} did{tradeDay ? ` on ${ukDay(tradeDay)}` : ""}</h2>
          <p className="lede">{EXPECT.strategies[maestroKey]?.desc}</p>
        </div>
        {dayTrades.length ? (
          <ol className={`panel ${styles.log}`}>
            {dayTrades.map((t) => (
              <li key={t.opened}>
                <span className={`num ${styles.time}`}>{ukTime(t.opened)}</span>
                <span>
                  Expected the euro to {t.direction > 0 ? "rise, so bought euros" : "fall, so sold euros"}.{" "}
                  {t.open
                    ? `Still holding: ${money(t.gross_gbp)} so far.`
                    : `Closed it at ${ukTime(t.closed)}: ${t.gross_gbp >= 0 ? "made" : "lost"} ${money(Math.abs(t.gross_gbp), false)}, paid ${money(t.cost_gbp, false)} in fees.`}
                </span>
              </li>
            ))}
          </ol>
        ) : (
          <p className={`panel ${styles.notice}`}>
            No trades yet. It only trades when it expects a move worth more than the fee.
          </p>
        )}
        <p className={styles.small}>Times are UK time. Pounds are on the £10,000 of pretend money.</p>
      </section>

      {/* 5. Real vs pretend */}
      <section className={styles.section} aria-labelledby="real">
        <div className={styles.sectionHead}>
          <h2 id="real">Pretend trades against real ones</h2>
          {o.filled > 0 ? (
            <p className="lede">
              One copy of {label(maestroKey)} also places real orders with a broker, on a practice account.
              Its <strong>{o.filled}</strong> real trades got prices{" "}
              {(o.mean_extra_pips ?? 0) >= 0 ? "worse" : "better"} than the pretend ones by{" "}
              <strong>{pips(Math.abs(o.mean_extra_pips ?? 0), 2)} pips</strong> on average, which adds up
              to {money(Math.abs(o.total_extra_gbp ?? 0), false)} {(o.total_extra_gbp ?? 0) >= 0 ? "more" : "less"} in
              costs. Orders reached the broker {o.median_delay_s ?? "n/a"} seconds after each decision.
            </p>
          ) : (
            <p className="lede">
              When the trial proper starts, one copy of MAESTRO will also place real orders with a broker,
              on a practice account with no real money. This section will compare every real order with
              its pretend twin: the gap is what a backtest can&rsquo;t see.
            </p>
          )}
        </div>
        {o.filled + o.failed > 0 && (
          <details className={styles.more}>
            <NumbersToggle />
            <div className={styles.moreBody}>
              <p>
                &ldquo;Against paper&rdquo; is how far the fill was from the mid price at the decision, less
                the half-spread the paper ledger already charges; positive means reality cost more. Delay
                runs from the bar&rsquo;s close to the broker&rsquo;s fill. {o.failed} orders failed.
              </p>
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead><tr><th>Decision (UTC)</th><th>Units</th><th>Status</th><th>From mid</th><th>Against paper</th><th>Spread</th><th>Delay</th></tr></thead>
                  <tbody>
                    {o.recent.slice(-30).reverse().map((r) => (
                      <tr key={r.bar}>
                        <td className="num">{r.bar.slice(0, 16).replace("T", " ")}</td>
                        <td className="num">{r.units.toLocaleString("en-GB")}</td>
                        <td>{r.status}</td>
                        <td className="num">{pips(r.vs_mid_pips, 2)}</td>
                        <td className="num">{pips(r.extra_pips, 2)}</td>
                        <td className="num">{pips(r.spread_pips, 2)}</td>
                        <td className="num">{r.delay_s === null ? "n/a" : `${r.delay_s}s`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        )}
      </section>

      {/* 6. Health and audit trail */}
      <section className={styles.section} aria-labelledby="check">
        <div className={styles.sectionHead}>
          <h2 id="check">Is it running, and can you check it?</h2>
          <p className="lede">
            The system has checked the market <strong>{(h.bars_recorded ?? 0).toLocaleString("en-GB")}</strong>{" "}
            times since the start and missed <strong>{h.bars_missed ?? 0}</strong>. Everything on this
            page can be checked against the published record.
          </p>
        </div>
        <ul className={`panel ${styles.facts}`}>
          <li><span>Snapshots</span><a href={LIVE_REPO} target="_blank" rel="noreferrer">every night&rsquo;s data, never rewritten</a></li>
          <li><span>The forecast it&rsquo;s judged against</span><a href={`${REPO}/blob/main/web/src/data/live_expectations.json`} target="_blank" rel="noreferrer">written {longDate(EXPECT.created_at)}</a></li>
          <li><span>Code running now</span><a href={commitUrl(snap.code_commit)} target="_blank" rel="noreferrer" className="num">{snap.code_commit?.slice(0, 7) ?? "unknown"}</a></li>
          <li><span>MAESTRO&rsquo;s training data</span><span className="num">{snap.model.train_start?.slice(0, 10)} to {snap.model.train_end?.slice(0, 10)}</span></li>
          <li><span>Bars checked</span><span className="num">{h.bars_recorded ?? 0} of {h.bars_expected ?? 0} ({h.bars_unscorable ?? 0} without a forecast)</span></li>
          <li><span>How it is scored</span><Link href="/method">the same rules as the 20-year test</Link></li>
        </ul>
        <p className={styles.small}>Pretend money and an OANDA practice account only. Nothing here is investment advice.</p>
      </section>
    </div>
  );
}
