import type { Metadata } from "next";
import Link from "next/link";
import { MSC, RESULTS } from "@/lib/data";
import { pct } from "@/lib/format";
import styles from "./method.module.css";

export const metadata: Metadata = {
  title: "How it was tested",
  description:
    "The walk-forward design, cost model, timing rules and automated checks behind every number on the MAESTRO site.",
};

const REPO = "https://github.com/Fran6jy/maestro/blob/main";

const SECTIONS = [
  { id: "data", label: "The data" },
  { id: "walk-forward", label: "Walk-forward testing" },
  { id: "costs", label: "Costs" },
  { id: "timing", label: "No peeking" },
  { id: "metrics", label: "What each number means" },
  { id: "strategies", label: "Strategy settings" },
  { id: "msc", label: "The MSc figures" },
  { id: "limits", label: "Limitations" },
  { id: "reproduce", label: "Reproduce it" },
];

function WalkForward() {
  const rows = 7;
  const W = 720;
  const rowH = 30;
  const H = rows * rowH + 34;
  const left = 110;
  const unit = (W - left - 16) / (rows + 6);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.wf} role="img"
      aria-label="Each test month is traded only after training on all earlier data, with a five-day gap between training and testing.">
      {Array.from({ length: rows }).map((_, i) => {
        const y = 10 + i * rowH;
        const trainW = unit * (5 + i);
        return (
          <g key={i}>
            <text x={0} y={y + 15} className={styles.wfLabel}>Window {i + 1}</text>
            <rect x={left} y={y} width={trainW} height={18} rx={3} className={styles.wfTrain} />
            <rect x={left + trainW + 3} y={y} width={5} height={18} className={styles.wfGap} />
            <rect x={left + trainW + 11} y={y} width={unit - 2} height={18} rx={3} className={styles.wfTest} />
          </g>
        );
      })}
      <g transform={`translate(${left}, ${rows * rowH + 20})`}>
        <rect width={14} height={10} rx={2} className={styles.wfTrain} />
        <text x={20} y={9} className={styles.wfKey}>Training data (grows each month)</text>
        <rect x={270} width={6} height={10} className={styles.wfGap} />
        <text x={282} y={9} className={styles.wfKey}>5-day gap</text>
        <rect x={370} width={14} height={10} rx={2} className={styles.wfTest} />
        <text x={390} y={9} className={styles.wfKey}>Test month, traded blind</text>
      </g>
    </svg>
  );
}

export default function MethodPage() {
  const first = RESULTS.dates[0];
  const last = RESULTS.dates[RESULTS.dates.length - 1];

  return (
    <div className={`shell ${styles.page}`}>
      <header className={styles.head}>
        <p className="eyebrow">Methodology</p>
        <h1>How it was tested</h1>
        <p className="lede">
          Every number on this site comes from one piece of scoring code, run the same way for every
          strategy. This page explains the rules, so anyone can check them or run them again.
        </p>
      </header>

      <div className={styles.layout}>
        <nav className={styles.toc} aria-label="On this page">
          <p className="eyebrow">On this page</p>
          <ol>
            {SECTIONS.map((s) => (
              <li key={s.id}><a href={`#${s.id}`}>{s.label}</a></li>
            ))}
          </ol>
        </nav>

        <article className={styles.article}>
          <section id="data">
            <h2>The data</h2>
            <p>
              OANDA EUR/USD mid prices in 5-minute bars, from January 2022 to March 2026. The first
              year is only ever used for training. Results cover {first} to {last}:{" "}
              {RESULTS.dates.length.toLocaleString("en-GB")} trading days across {RESULTS.months.length}{" "}
              monthly test windows.
            </p>
          </section>

          <section id="walk-forward">
            <h2>Walk-forward testing</h2>
            <p>
              A strategy is trained on everything before a test month, then trades that month without
              having seen it. The window then moves forward one month and the process repeats, 39 times.
              A five-day gap separates training from testing, so no training label can overlap a test
              price.
            </p>
            <figure className={`panel ${styles.figure}`}>
              <WalkForward />
            </figure>
          </section>

          <section id="costs">
            <h2>Costs</h2>
            <p>
              Every round trip (buying then selling, or the reverse) pays{" "}
              <strong className="num">{RESULTS.refCostPips.toFixed(1)} pips</strong>, the typical EUR/USD
              spread in MAESTRO&rsquo;s cost model. Trades are one unit of EUR/USD with no leverage. The
              Strategy Lab&rsquo;s slider re-prices the same trades at any cost from 0 to 2 pips; because
              cost scales linearly with the number of trades, this is exact rather than an estimate.
            </p>
          </section>

          <section id="timing">
            <h2>No peeking</h2>
            <p>
              A decision made at the close of one bar can only act on the next bar. Every test window
              starts flat and ends flat, so every trade pays exactly one round-trip cost. Automated
              tests check that:
            </p>
            <ul>
              <li>a strategy that secretly reads the next bar scores 100%, proving the scorer would catch it;</li>
              <li>a signal built from the current bar&rsquo;s own move scores about 50% on random data;</li>
              <li>the regression models only ever learn from their training window.</li>
            </ul>
            <p>
              See <a href={`${REPO}/tests/test_baselines.py`} target="_blank" rel="noreferrer">tests/test_baselines.py</a>.
            </p>
          </section>

          <section id="metrics">
            <h2>What each number means</h2>
            <dl className={styles.defs}>
              <dt>Right about the next move</dt>
              <dd>Of the bars where a strategy held a position and the price moved, the share where the position was on the right side.</dd>
              <dt>Months in profit</dt>
              <dd>Test months that ended with more pips than they started, after costs.</dd>
              <dt>Sharpe ratio</dt>
              <dd>Average daily return divided by its variability, scaled to a year (×√252). Above 1 is usually considered good.</dd>
              <dt>Breaks even at</dt>
              <dd>The average profit per trade before costs, in pips. A broker charging more than this makes the strategy lose money.</dd>
              <dt>Balance from £10,000</dt>
              <dd>What a starting £10,000 becomes if every day&rsquo;s return compounds, at the chosen cost.</dd>
            </dl>
          </section>

          <section id="strategies">
            <h2>Strategy settings</h2>
            <p>
              The MSc report doesn&rsquo;t state every parameter, so standard defaults fill the gaps.
              Moving averages use 20 and 200 bars. The contrarian strategy looks back 3 bars. Bollinger
              bands use 20 bars at 2 standard deviations. The regressions use the latest 1 or 5 price
              moves. Buy and hold buys at the start of each test month and sells at the end.
            </p>
          </section>

          <section id="msc">
            <h2>The MSc figures</h2>
            <p>
              The MSc reported a {MSC.reported_hit.toFixed(2)}% hit ratio for a one-move linear regression,
              trained and scored on the same month of 5-minute prices (29 June to 31 July 2023), and a
              Sharpe ratio of {MSC.reported_sharpe} for a separate daily moving-average strategy from 2014
              to 2023. Re-running the regression on the same month gives {pct(MSC.rerun["5dp"].hit)} with
              5-decimal prices and {pct(MSC.rerun["4dp"].hit)} when prices are rounded to 4 decimals. The
              original notebook is lost, so the exact data source can&rsquo;t be confirmed.
            </p>
          </section>

          <section id="limits">
            <h2>Limitations</h2>
            <ul>
              <li>One currency pair so far. Other markets may behave differently.</li>
              <li>Costs are a fixed spread per trade. Real spreads widen around news and overnight.</li>
              <li>The MSc ran its moving-average strategy on daily bars; here it runs on 5-minute bars.</li>
              <li>This is a historical simulation. The live practice-account trial will measure how far reality differs.</li>
            </ul>
          </section>

          <section id="reproduce">
            <h2>Reproduce it</h2>
            <p>From a copy of the repository, with the price data in place:</p>
            <pre className={styles.code}><code>{`python -m maestro.backtesting.baselines
python -m pytest maestro/tests/test_baselines.py
python -m maestro.demo.export_web_data`}</code></pre>
            <p>
              The last command rebuilds the data this site is made from. Source:{" "}
              <a href={`${REPO}/backtesting/baselines.py`} target="_blank" rel="noreferrer">backtesting/baselines.py</a>.
            </p>
            <p className={styles.back}>
              <Link href="/#lab" className="btn btn-ghost">Back to the Strategy Lab</Link>
            </p>
          </section>
        </article>
      </div>
    </div>
  );
}
