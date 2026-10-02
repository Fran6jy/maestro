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
    body: "20 years of walk-forward months, one shared scorer, macro data used only once published, automated checks that no strategy sees the future, and a sealed final period, opened once on 2 October 2026: same answer, every strategy inside its pre-registered range.",
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
    status: ["pill-next", "Running"],
    title: "Live practice trial",
    body: "Since 2 October 2026, MAESTRO and the MSc strategies trade side by side on every 5-minute bar, paper-traded at live OANDA prices, and MAESTRO's top-10% version also places orders on a practice account. Judged against ranges published before it started. No real money.",
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
        <ol className={styles.list}>
          {ITEMS.map((it, i) => (
            <li key={it.title} className={styles.item} data-state={it.status[0].replace("pill-", "")}>
              <span className={styles.num} aria-hidden="true">{String(i + 1).padStart(2, "0")}</span>
              <div className={styles.text}>
                <p className={styles.state}>{it.status[1]}</p>
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
