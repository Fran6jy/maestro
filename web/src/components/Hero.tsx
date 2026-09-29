import Link from "next/link";
import RidgeCanvas from "./RidgeCanvas";
import { RESULTS, RIDGE, STRATEGY } from "@/lib/data";
import { finalBalance } from "@/lib/metrics";
import { gbp, pct, signed } from "@/lib/format";
import styles from "./Hero.module.css";

export default function Hero() {
  const lr = STRATEGY.logreg_lag5;
  const before = finalBalance(lr, 0);
  const after = finalBalance(lr, RESULTS.refCostPips);

  const stats = [
    { value: pct(lr.hit), label: "how often a simple model calls the next 5-minute move", tone: "" },
    { value: `${signed(lr.grossPerTrade)} pips`, label: "what it earns per trade before costs", tone: styles.amber },
    { value: `${RESULTS.refCostPips.toFixed(1)} pips`, label: "what every trade costs in spread", tone: styles.cost },
    { value: `${RESULTS.months.length} months`, label: "tested one at a time, never seen in advance", tone: "" },
  ];

  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <div className={styles.visual}>
        <RidgeCanvas paths={RIDGE.paths} days={RIDGE.days} />
        <div className={styles.glow} aria-hidden="true" />
      </div>

      <div className={`shell ${styles.content}`}>
        <p className={`eyebrow ${styles.kicker}`}>
          <span className={styles.dot} aria-hidden="true" />
          MAESTRO · PhD research · EUR/USD
        </p>
        <h1 id="hero-title" className={styles.title}>
          The cost of <em>being right.</em>
        </h1>
        <p className={styles.sub}>
          Can a team of AI agents trade currencies profitably once every trade pays its way? I rebuilt
          the strategies from my MSc and tested them fairly on three years of EUR/USD. £10,000 on a
          simple logistic-regression model becomes <strong className={styles.amberText}>{gbp(before)}</strong> before
          costs and <strong className={styles.coolText}>{gbp(after)}</strong> after.
        </p>
        <div className={styles.actions}>
          <Link href="#lab" className="btn btn-primary">
            Try the Strategy Lab <span className="arrow" aria-hidden="true">→</span>
          </Link>
          <Link href="/method" className="btn btn-ghost">
            How it was tested
          </Link>
        </div>
      </div>

      <div className={`shell ${styles.statsWrap}`}>
        <dl className={styles.stats}>
          {stats.map((s) => (
            <div key={s.label}>
              <dt>{s.label}</dt>
              <dd className={`num ${s.tone}`}>{s.value}</dd>
            </div>
          ))}
        </dl>
        <p className={styles.caption}>
          Behind this text: {RIDGE.paths.length} real EUR/USD trading days, midnight to midnight UTC.
          Each ridge is one day. Point at one to read it.
        </p>
      </div>
    </section>
  );
}
