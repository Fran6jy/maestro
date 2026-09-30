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
  const rows = 6;
  const W = 720;
  const rowH = 30;
  const H = rows * rowH + 34;
  const left = 110;
  const unit = (W - left - 16) / (rows + 4);   // one unit = three months
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.wf} role="img"
      aria-label="Every three months the models are retrained on the latest twelve months of data, then trade the next three months blind, with a five-day gap between training and testing.">
      {Array.from({ length: rows }).map((_, i) => {
        const y = 10 + i * rowH;
        const trainX = left + unit * i;
        const trainW = unit * 4;
        return (
          <g key={i}>
            <text x={0} y={y + 15} className={styles.wfLabel}>Retrain {i + 1}</text>
            <rect x={trainX} y={y} width={trainW} height={18} rx={3} className={styles.wfTrain} />
            <rect x={trainX + trainW + 3} y={y} width={5} height={18} className={styles.wfGap} />
            <rect x={trainX + trainW + 11} y={y} width={unit - 2} height={18} rx={3} className={styles.wfTest} />
          </g>
        );
      })}
      <g transform={`translate(${left}, ${rows * rowH + 20})`}>
        <rect width={14} height={10} rx={2} className={styles.wfTrain} />
        <text x={20} y={9} className={styles.wfKey}>Latest 12 months of training data</text>
        <rect x={270} width={6} height={10} className={styles.wfGap} />
        <text x={282} y={9} className={styles.wfKey}>5-day gap</text>
        <rect x={370} width={14} height={10} rx={2} className={styles.wfTest} />
        <text x={390} y={9} className={styles.wfKey}>3 test months, traded blind</text>
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
              OANDA EUR/USD prices in 5-minute bars, with bid and ask, from January 2005. The first
              year is only ever used for training, and everything from 7 March 2026 is sealed for one
              final confirmation test, so results cover {first} to {last}:{" "}
              {RESULTS.dates.length.toLocaleString("en-GB")} trading days across {RESULTS.months.length}{" "}
              monthly test windows. Macro data (VIX, US yields, interest rates and inflation) comes from
              FRED and reaches a bar only once it had been published.
            </p>
          </section>

          <section id="walk-forward">
            <h2>Walk-forward testing</h2>
            <p>
              Every three months, each model is retrained on the latest twelve months of data, then
              trades the next three months without having seen them. The window then moves forward
              three months and the process repeats: {Math.ceil(RESULTS.months.length / 3)} retrainings,{" "}
              {RESULTS.months.length} test months, each scored on its own. A five-day gap separates training from testing, so no training label can overlap a
              test price. Rules-based strategies such as the moving-average crossover have nothing to
              train, so they simply trade every month.
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
              OANDA&rsquo;s actual quoted spread, measured on every bar, had a median of 0.9&ndash;1.6 pips
              depending on the year (1.5&ndash;1.6 since 2022), so 0.8 pips is on the generous side.
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
              <li>the regression models only ever learn from their training window;</li>
              <li>no retraining ever uses data from the months it is about to trade;</li>
              <li>macro data reaches a bar only once it was published;</li>
              <li>market-regime labels use only the bars up to each bar, so rewriting later prices
                cannot change an earlier label;</li>
              <li>MAESTRO&rsquo;s confidence cut-off only uses confidence from earlier bars.</li>
            </ul>
            <p>
              Two of these checks exist because of bugs found on 30 September 2026. Macro values had
              been stamped at midnight on the day they describe, so each bar saw that day&rsquo;s closing
              VIX, and inflation six weeks before it was published. And MAESTRO&rsquo;s regime detector
              labelled each bar using the whole three-month test block around it, so a label could depend
              on later prices. Both are fixed; MAESTRO&rsquo;s earlier results are withdrawn and being
              re-run. The other strategies use prices only and were not affected.
            </p>
            <p>
              See <a href={`${REPO}/tests`} target="_blank" rel="noreferrer">the tests</a>.
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
            <p>
              MAESTRO forecasts the next 30 minutes with its regime, TFT and PatchTST agents, retrained
              on the same schedule, with two months of each training window held back to stop training
              early. Its results are being re-run after the fixes above. How forecasts become trades is a design choice, so three versions
              were fixed before any result was seen: <strong>as designed</strong> (trade only above the
              confidence thresholds MAESTRO was built with), <strong>most confident 10%</strong> (the
              cut-off taken from the previous five trading days), and <strong>every signal</strong>.
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
              <li>Costs are a fixed spread per trade. Real spreads are wider on average and widen further
                around news and overnight.</li>
              <li>The MSc ran its moving-average strategy on daily bars; here it runs on 5-minute bars.</li>
              <li>MAESTRO&rsquo;s model settings were not tuned. Tuning them on these months would
                have let the test leak into the design.</li>
              <li>This is a historical simulation. A live trial on an OANDA practice account, now in a
                paper-only shakedown, will measure how far reality differs.</li>
            </ul>
          </section>

          <section id="reproduce">
            <h2>Reproduce it</h2>
            <p>
              OANDA&rsquo;s prices may not be redistributed, so the repository has none. With your own
              OANDA practice account and a free FRED key:
            </p>
            <pre className={styles.code}><code>{`python -m maestro.data.pipeline.store fetch --start 2005-01-01
python -m maestro.data.pipeline.store sync
python -m maestro.backtesting.baselines --refit-months 3 --train-months 12
python -m pytest maestro/tests --ignore=maestro/tests/smoke_test.py
python -m maestro.demo.export_web_data`}</code></pre>
            <p>
              The first two build the price and macro store and the features; the third scores every
              baseline over 20 years; the last rebuilds the data this site is made from. Source:{" "}
              <a href={`${REPO}/backtesting/baselines.py`} target="_blank" rel="noreferrer">backtesting/baselines.py</a>{" "}
              and <a href={`${REPO}/backtesting/maestro_runner.py`} target="_blank" rel="noreferrer">backtesting/maestro_runner.py</a>.
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
