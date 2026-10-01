import styles from "./Thesis.module.css";

const STEPS = [
  {
    n: "01",
    title: "Rebuild",
    body: "Re-create every strategy from my MSc, because its headline figures could not be reproduced.",
    status: { cls: "pill-done", label: "Done" },
  },
  {
    n: "02",
    title: "Measure",
    body: "Score them all with one piece of code, month by month, never letting a strategy see the future, and charging for every trade.",
    status: { cls: "pill-done", label: "Done" },
  },
  {
    n: "03",
    title: "Challenge",
    body: "Put MAESTRO, a team of specialist AI agents, through exactly the same test. It calls more moves right than anything else, and still doesn't beat the costs.",
    status: { cls: "pill-done", label: "Done" },
  },
];

export default function Thesis() {
  return (
    <section className="band" aria-labelledby="thesis-title">
      <div className="shell">
        <blockquote className={styles.quote} id="thesis-title">
          My MSc asked the right questions but couldn&rsquo;t answer them rigorously. This PhD rebuilds
          the same strategies under a leakage-free, cost-aware test, shows what they{" "}
          <span>actually achieve</span>, then asks whether a multi-agent system does better.
        </blockquote>
        <ol className={styles.steps}>
          {STEPS.map((s) => (
            <li key={s.n}>
              <div className={styles.stepHead}>
                <span className={`mono ${styles.n}`}>{s.n}</span>
                <span className={`pill ${s.status.cls}`}>{s.status.label}</span>
              </div>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
