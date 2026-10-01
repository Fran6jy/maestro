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
    body: "20 years of walk-forward months, one shared scorer, macro data used only once published, a sealed final test period, and automated checks that no strategy sees the future.",
  },
  {
    status: ["pill-done", "Done"],
    title: "The baselines",
    body: "Over 20 years the models that learn from past moves call the next move right about 52% of the time, a real edge, but it earns under a pip per trade. Every strategy loses money after costs.",
  },
  {
    status: ["pill-done", "Done"],
    title: "MAESTRO vs the baselines",
    body: "Same months, costs and scoring code. MAESTRO calls 53–54% of moves right, the best of any strategy, but earns about a third of a trade's cost, so it loses money too. Re-run after a look-ahead bug fix, which barely changed it.",
  },
  {
    status: ["pill-next", "Shakedown"],
    title: "Live practice trial",
    body: "MAESTRO and the MSc strategies trade side by side on every 5-minute bar, paper-traded at live OANDA prices, and one MAESTRO version will also place orders on a practice account. Running as a paper-only shakedown now; the trial proper starts after a final test on data no model has seen. No real money.",
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
          <h2 id="roadmap-title">Where the research is now.</h2>
          <p className="lede">
            A clear &ldquo;no&rdquo; is a valid result. Each step is only reported once it has passed the
            same checks as the one before it.
          </p>
        </div>
        <ol className={styles.track}>
          {ITEMS.map((it, i) => (
            <li key={it.title} className={`reveal ${styles.item}`} data-state={it.status[0].replace("pill-", "")}>
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
