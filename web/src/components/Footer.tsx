import Link from "next/link";
import Logo from "./Logo";
import { RESULTS } from "@/lib/data";
import styles from "./Footer.module.css";

const PERIOD = `${RESULTS.months[0].label} to ${RESULTS.months[RESULTS.months.length - 1].label}`;

const REPO = "https://github.com/Fran6jy/maestro";

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`shell ${styles.grid}`}>
        <div className={styles.brand}>
          <Logo />
          <p>
            PhD research at Coventry University into whether a team of AI agents can trade EUR/USD
            profitably once every trade pays its costs.
          </p>
        </div>
        <nav className={styles.cols} aria-label="Footer">
          <div>
            <p className="eyebrow">Explore</p>
            <ul>
              <li><Link href="/#lab">Strategy Lab</Link></li>
              <li><Link href="/#msc">The MSc, re-examined</Link></li>
              <li><Link href="/#agents">The agents</Link></li>
              <li><Link href="/#roadmap">Roadmap</Link></li>
            </ul>
          </div>
          <div>
            <p className="eyebrow">Research</p>
            <ul>
              <li><Link href="/method">How it was tested</Link></li>
              <li><a href={REPO} target="_blank" rel="noreferrer">Source code</a></li>
              <li><a href={`${REPO}/blob/main/backtesting/baselines.py`} target="_blank" rel="noreferrer">The shared scorer</a></li>
            </ul>
          </div>
        </nav>
      </div>
      <div className={`shell ${styles.base}`}>
        <p>
          Research, not investment advice. The results here come from historical simulation and a
          practice account with pretend money. No real money is traded.
        </p>
        <p className="mono">EUR/USD · 5-minute bars · {PERIOD}</p>
      </div>
    </footer>
  );
}
