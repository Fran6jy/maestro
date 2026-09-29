import styles from "./Roadmap.module.css";

const ITEMS = [
  {
    status: ["pill-done", "Done"],
    title: "The MSc, re-examined",
    body: "Its headline figures were traced to two unrelated tests and re-run. They don't reproduce.",
  },
  {
    status: ["pill-done", "Done"],
    title: "A fair test",
    body: "39 walk-forward months, one shared scorer, and automated checks that no strategy sees the future.",
  },
  {
    status: ["pill-done", "Done"],
    title: "The baselines",
    body: "Every MSc strategy loses money after costs. Buy and hold is the only positive result.",
  },
  {
    status: ["pill-next", "Next"],
    title: "MAESTRO vs the baselines",
    body: "The six-agent system goes through the same months, costs and scoring code. It has to beat them after costs.",
  },
  {
    status: ["pill-plan", "Planned"],
    title: "Live practice trial",
    body: "MAESTRO and the MSc strategies trade side by side on an OANDA practice account, checked daily against the backtest. No real money.",
  },
  {
    status: ["pill-later", "Later"],
    title: "Beyond EUR/USD",
    body: "Gold and the S&P 500 under one shared market-regime layer, if the single-currency results justify it.",
  },
] as const;

export default function Roadmap() {
  return (
    <section id="roadmap" className="band" aria-labelledby="roadmap-title">
      <div className="shell">
        <div className="band-head">
          <p className="eyebrow">Roadmap</p>
          <h2 id="roadmap-title">Where the research is now.</h2>
          <p className="lede">
            A clear &ldquo;no&rdquo; is a valid result. Each step is only reported once it has passed the
            same checks as the one before it.
          </p>
        </div>
        <ol className={styles.track}>
          {ITEMS.map((it, i) => (
            <li key={it.title} className={`reveal ${styles.item}`} data-state={it.status[1].toLowerCase()}>
              <div className={styles.marker} aria-hidden="true">
                <span className={styles.dot} />
                {i < ITEMS.length - 1 && <span className={styles.rail} />}
              </div>
              <div className={styles.body}>
                <span className={`pill ${it.status[0]}`}>{it.status[1]}</span>
                <h3>{it.title}</h3>
                <p>{it.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
