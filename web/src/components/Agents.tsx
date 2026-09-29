"use client";

import { useState } from "react";
import styles from "./Agents.module.css";

type AgentId = "regime" | "signal" | "sentiment" | "orchestrator" | "risk" | "execution";

interface Agent {
  id: AgentId;
  role: string;
  name: string;
  tech: string;
  does: string;
  hands: string;
  x: number;
  y: number;
}

const AGENTS: Agent[] = [
  {
    id: "regime", role: "Weather reporter", name: "Regime agent",
    tech: "Hidden Markov model + Transformer",
    does: "Works out what kind of market this is: calm, trending, sideways or in crisis. Every other agent hears this first, and the risk agent sizes trades down when the weather turns.",
    hands: "The market regime, and how sure it is",
    x: 9, y: 50,
  },
  {
    id: "signal", role: "Analyst", name: "Signal agent",
    tech: "Temporal Fusion Transformer + PatchTST",
    does: "Forecasts the next 30 minutes of EUR/USD and says how confident it is. Its forecasting style shifts with the regime it is told about.",
    hands: "Buy, sell or wait, with a confidence",
    x: 30, y: 20,
  },
  {
    id: "sentiment", role: "News reader", name: "Sentiment agent",
    tech: "FinBERT + GPT-4o",
    does: "Reads central-bank and economic headlines and scores whether they favour the euro or the dollar, tracking how often its reading has been right.",
    hands: "The tone of the news, and how reliable it has been",
    x: 30, y: 80,
  },
  {
    id: "orchestrator", role: "Head of the desk", name: "Meta-orchestrator",
    tech: "Regime-weighted fusion, with an optional LLM reasoning layer",
    does: "Weighs the analyst against the news reader in light of the regime, settles disagreements and makes the call. It records why, in plain language, for every decision.",
    hands: "One decision, with the reasons behind it",
    x: 51, y: 50,
  },
  {
    id: "risk", role: "Safety officer", name: "Risk agent",
    tech: "CVaR limits + Kelly sizing + reinforcement learning",
    does: "Decides how much to risk. It can shrink a trade or veto it outright, and no other agent can overrule it.",
    hands: "Position size, stop-loss and take-profit",
    x: 72, y: 50,
  },
  {
    id: "execution", role: "Trader", name: "Execution agent",
    tech: "Cost-aware order placement",
    does: "Places the order and records what it really cost, spread and slippage included. Those real costs are what the Strategy Lab above charges for.",
    hands: "Filled trades and their actual cost",
    x: 91, y: 50,
  },
];

const EDGES: [AgentId, AgentId][] = [
  ["regime", "signal"],
  ["regime", "sentiment"],
  ["regime", "orchestrator"],
  ["signal", "orchestrator"],
  ["sentiment", "orchestrator"],
  ["orchestrator", "risk"],
  ["risk", "execution"],
];

const BY_ID = Object.fromEntries(AGENTS.map((a) => [a.id, a])) as Record<AgentId, Agent>;

function edgePath(a: Agent, b: Agent) {
  const mx = (a.x + b.x) / 2;
  return `M${a.x} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${b.x} ${b.y}`;
}

export default function Agents() {
  const [sel, setSel] = useState<AgentId>("orchestrator");
  const agent = BY_ID[sel];
  const connected = new Set<AgentId>([sel]);
  EDGES.forEach(([a, b]) => {
    if (a === sel) connected.add(b);
    if (b === sel) connected.add(a);
  });

  return (
    <section id="agents" className="band" aria-labelledby="agents-title">
      <div className="shell">
        <div className="band-head">
          <p className="eyebrow">Inside MAESTRO</p>
          <h2 id="agents-title">Six specialists, one desk.</h2>
          <p className="lede">
            A single model has to be good at everything. MAESTRO splits the job the way a trading desk
            does, so each piece can be tested, switched off and measured on its own. Select an agent to
            see what it does and who it reports to.
          </p>
        </div>

        <div className={`panel ${styles.room}`}>
          <div className={styles.diagram}>
            <svg className={styles.edges} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {EDGES.map(([a, b]) => {
                const on = a === sel || b === sel;
                return (
                  <g key={a + b}>
                    <path d={edgePath(BY_ID[a], BY_ID[b])} className={`${styles.edge} ${on ? styles.edgeOn : ""}`} vectorEffect="non-scaling-stroke" />
                    <path d={edgePath(BY_ID[a], BY_ID[b])} className={`${styles.flow} ${on ? styles.flowOn : ""}`} vectorEffect="non-scaling-stroke" />
                  </g>
                );
              })}
            </svg>
            <div className={styles.nodes} role="radiogroup" aria-label="Agents">
              {AGENTS.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  role="radio"
                  aria-checked={a.id === sel}
                  className={`${styles.node} ${a.id === sel ? styles.nodeOn : ""} ${connected.has(a.id) ? "" : styles.nodeDim}`}
                  style={{ "--x": `${a.x}%`, "--y": `${a.y}%` } as React.CSSProperties}
                  onClick={() => setSel(a.id)}
                >
                  <span className={styles.nodeRole}>{a.role}</span>
                  <span className={styles.nodeName}>{a.name}</span>
                </button>
              ))}
            </div>
          </div>

          <div className={styles.detail} aria-live="polite">
            <div className={styles.detailHead}>
              <p className="eyebrow">{agent.name}</p>
              <h3>{agent.role}</h3>
            </div>
            <p className={styles.does}>{agent.does}</p>
            <dl className={styles.meta}>
              <div>
                <dt>Under the hood</dt>
                <dd>{agent.tech}</dd>
              </div>
              <div>
                <dt>Hands on</dt>
                <dd>{agent.hands}</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd><span className="pill pill-next">Built · being tested next</span></dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </section>
  );
}
