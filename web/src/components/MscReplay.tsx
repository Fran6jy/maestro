"use client";

import { useEffect, useRef, useState } from "react";
import { MSC } from "@/lib/data";
import { pct } from "@/lib/format";
import styles from "./MscReplay.module.css";

type Dec = "5dp" | "4dp";

function Gauge({ hit }: { hit: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(560);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const H = 118;
  const pad = 18;
  const axisY = 66;
  const lo = 0.25;
  const hi = 0.55;
  const X = (v: number) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
  const narrow = W < 460;
  const ticks = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55];

  const marks = [
    { v: MSC.reported_hit / 100, label: `${narrow ? "MSc" : "MSc reported"} ${MSC.reported_hit.toFixed(1)}%`, color: "var(--cost)", y: axisY - 16 },
    { v: hit, label: `Re-run ${pct(hit)}`, color: "var(--amber)", y: axisY - 44 },
    { v: 0.5, label: "Coin flip", color: "var(--ink-3)", y: axisY + 44 },
  ];

  return (
    <div ref={ref} className={styles.gauge}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`MSc reported ${MSC.reported_hit}%. Re-run scores ${pct(hit)}. A coin flip scores 50%.`}>
        <line x1={X(lo)} x2={X(hi)} y1={axisY} y2={axisY} className={styles.axisLine} />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={axisY - 4} y2={axisY + 4} className={styles.tick} />
            <text x={X(t)} y={axisY + 22} textAnchor="middle" className={styles.tickLabel}>
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {marks.map((mk) => {
          const x = X(mk.v);
          const half = mk.label.length * 3.8;
          const anchor = x - half < 0 ? "start" : x + half > W ? "end" : "middle";
          return (
            <g key={mk.label} className={styles.mark} style={{ transform: `translateX(0)` }}>
              {mk.y < axisY - 24 && <line x1={x} x2={x} y1={axisY - 8} y2={mk.y + 5} stroke={mk.color} strokeWidth={1.5} />}
              <circle cx={x} cy={axisY} r={7} fill={mk.color} />
              <text x={x} y={mk.y} textAnchor={anchor} fill={mk.color} className={styles.markLabel}>
                {mk.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Strip({ sample }: { sample: number[] }) {
  const W = 960;
  const H = 70;
  const n = sample.length;
  const step = W / n;
  const bw = step * 0.6;
  const mid = H / 2;
  const flats = sample.filter((v) => v === 0).length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.strip} role="img"
      aria-label={`${n} five-minute bars, ${flats} of which did not move`}>
      <line x1={0} x2={W} y1={mid} y2={mid} className={styles.mid} />
      {sample.map((v, i) => {
        const x = i * step + (step - bw) / 2;
        return v === 0 ? (
          <circle key={i} cx={x + bw / 2} cy={mid} r={3.2} className={styles.flat} />
        ) : (
          <rect key={i} x={x} width={bw} y={v > 0 ? mid - 27 : mid + 3} height={24} rx={1.5} className={styles.move} />
        );
      })}
    </svg>
  );
}

export default function MscReplay() {
  const [dec, setDec] = useState<Dec>("5dp");
  const r = MSC.rerun[dec];

  return (
    <section id="msc" className="band" aria-labelledby="msc-title">
      <div className={`shell ${styles.grid}`}>
        <div className={styles.copy}>
          <h2 id="msc-title">My MSc reported a 37.6% hit rate. It doesn&rsquo;t hold up.</h2>
          <p className="lede">
            That number came from <strong>one month</strong> of 5-minute prices, and the model was scored
            on the same data it learned from. The Sharpe ratio of 0.599 quoted next to it came from a
            different test altogether: a daily moving-average strategy from 2014 to 2023.
          </p>
          <p className="lede">
            Re-running the same regression on the same month gives <strong>{pct(MSC.rerun["5dp"].hit)}</strong>.
            The gap comes from how bars where the price doesn&rsquo;t move are scored: as wrong. Coarser
            prices mean more of those bars, and a lower score. The number measured the data feed as
            much as the model.
          </p>
          <dl className={styles.facts}>
            <div>
              <dt>MSc reported</dt>
              <dd className="num" style={{ color: "var(--cost)" }}>{MSC.reported_hit.toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Re-run, 5 decimals</dt>
              <dd className="num" style={{ color: "var(--amber)" }}>{pct(MSC.rerun["5dp"].hit)}</dd>
            </div>
            <div>
              <dt>Re-run, 4 decimals</dt>
              <dd className="num">{pct(MSC.rerun["4dp"].hit)}</dd>
            </div>
          </dl>
        </div>

        <figure className={`panel ${styles.demo}`}>
          <div className={styles.demoHead}>
            <div className={styles.toggle} role="group" aria-label="Price precision">
              {(["5dp", "4dp"] as Dec[]).map((d) => (
                <button key={d} type="button" aria-pressed={dec === d} onClick={() => setDec(d)}>
                  {d === "5dp" ? "Prices to 5 decimals" : "Prices to 4 decimals"}
                </button>
              ))}
            </div>
            <div className={styles.readout} aria-live="polite">
              <span className={`num ${styles.score}`}>{pct(r.hit)}</span>
              <span className={styles.scoreSub}>
                scored correct · <span className="num">{pct(r.flat_share)}</span> of bars flat
              </span>
            </div>
          </div>
          <Strip sample={MSC.sample[dec]} />
          <Gauge hit={r.hit} />
          <figcaption className={styles.cap}>
            The first eight hours of 29 June 2023, one bar every 5 minutes. Bars above the line are up
            moves and below are down moves. Grey dots are bars where the price didn&rsquo;t change,
            which the MSc&rsquo;s scoring counted as misses.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
