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

  // The thesis as one line of arithmetic.
  const net = lr.grossPerTrade - RESULTS.refCostPips;
  const terms = [
    { value: pct(lr.hit), label: "of next moves called right", tone: "" },
    { op: "→" },
    { value: signed(lr.grossPerTrade), label: "pips earned per trade", tone: styles.amber },
    { op: "−" },
    { value: RESULTS.refCostPips.toFixed(2), label: "pips every trade costs", tone: styles.cost },
    { op: "=" },
    { value: signed(net), label: "pips per trade, after costs", tone: styles.cool },
  ];

  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <div className={styles.visual}>
        <RidgeCanvas paths={RIDGE.paths} days={RIDGE.days} />
        <div className={styles.glow} aria-hidden="true" />
      </div>

      <div className={`shell ${styles.content}`}>
        <h1 id="hero-title" className={styles.title}>
          The cost of <em>being right.</em>
        </h1>
        <p className={styles.sub}>
          Can a team of AI agents trade currencies profitably once every trade pays its way? I rebuilt
          the strategies from my MSc and tested them fairly on {Math.round(RESULTS.months.length / 12)} years of EUR/USD. £10,000 on a
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
        <p className={styles.equation}
          aria-label={`A simple model calls ${pct(lr.hit)} of next moves right and earns ${signed(lr.grossPerTrade)} pips per trade; every trade costs ${RESULTS.refCostPips.toFixed(2)} pips, leaving ${signed(net)} pips per trade.`}>
          {terms.map((t, i) =>
            "op" in t ? (
              <span key={i} className={styles.op} aria-hidden="true">{t.op}</span>
            ) : (
              <span key={i} className={styles.term} aria-hidden="true">
                <span className={`${styles.termValue} ${t.tone}`}>{t.value}</span>
                <span className={styles.termLabel}>{t.label}</span>
              </span>
            ),
          )}
        </p>
        <p className={styles.caption}>
          Behind this text: {RIDGE.paths.length} real EUR/USD trading days, midnight to midnight UTC.
          Each ridge is one day. Point at one to read it.
        </p>
      </div>
    </section>
  );
}
