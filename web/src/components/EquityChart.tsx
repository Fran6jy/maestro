"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { gbp, niceDate, signed } from "@/lib/format";
import styles from "./EquityChart.module.css";

export interface ChartMonth {
  start: number;
  end: number;
  label: string;
}

interface Props {
  isoDates: string[];
  dates: number[];
  noCost: Float64Array;
  atCost: Float64Array;
  hold?: Float64Array;
  months: ChartMonth[];
  monthNet: number[];
  costLabel: string;
  title: string;
}

const Y_MIN = 1;
const Y_MAX = 30_000;
const TICKS = [1, 10, 100, 1_000, 10_000];
const TICK_LABEL: Record<number, string> = { 1: "£1", 10: "£10", 100: "£100", 1000: "£1k", 10000: "£10k" };

export default function EquityChart({
  isoDates, dates, noCost, atCost, hold, months, monthNet, costLabel, title,
}: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(900);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const narrow = W < 620;
  const H = narrow ? 320 : Math.round(Math.min(470, Math.max(360, W * 0.46)));
  const m = { l: narrow ? 42 : 52, r: narrow ? 62 : 86, t: 14, b: 66 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const x0 = dates[0];
  const x1 = dates[dates.length - 1];
  const ly0 = Math.log10(Y_MIN);
  const ly1 = Math.log10(Y_MAX);

  const X = (t: number) => m.l + ((t - x0) / (x1 - x0)) * pw;
  const Y = (v: number) => m.t + (1 - (Math.log10(Math.min(Y_MAX, Math.max(Y_MIN, v))) - ly0) / (ly1 - ly0)) * ph;

  const paths = useMemo(() => {
    const line = (arr: Float64Array) => {
      let d = "";
      for (let i = 0; i < arr.length; i++) d += (i ? "L" : "M") + X(dates[i]).toFixed(1) + " " + Y(arr[i]).toFixed(1);
      return d;
    };
    let gap = "";
    for (let i = 0; i < noCost.length; i++) gap += (i ? "L" : "M") + X(dates[i]).toFixed(1) + " " + Y(noCost[i]).toFixed(1);
    for (let i = atCost.length - 1; i >= 0; i--) gap += "L" + X(dates[i]).toFixed(1) + " " + Y(atCost[i]).toFixed(1);
    return { no: line(noCost), at: line(atCost), hold: hold ? line(hold) : "", gap: gap + "Z" };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W, noCost, atCost, hold]);

  const years = [2023, 2024, 2025, 2026].map((yr) => ({
    yr,
    s: Math.max(x0, Date.UTC(yr, 0, 1)),
    e: Math.min(x1, Date.UTC(yr + 1, 0, 1)),
  })).filter((y) => y.e > y.s);

  const last = noCost.length - 1;
  const ends = [
    { v: noCost[last], cls: styles.endNo },
    { v: atCost[last], cls: styles.endAt },
    ...(hold ? [{ v: hold[last], cls: styles.endHold }] : []),
  ].map((e) => ({ ...e, y: Y(e.v) + 4 })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;

  const stripY = m.t + ph + 34;

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const t = x0 + ((px - m.l) / pw) * (x1 - x0);
    let lo = 0;
    let hi = dates.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (dates[mid] < t) lo = mid;
      else hi = mid;
    }
    setHover(Math.abs(dates[lo] - t) < Math.abs(dates[hi] - t) ? lo : hi);
  };

  const hx = hover !== null ? X(dates[hover]) : 0;
  const tipLeft = hover !== null ? (hx + 14 + 190 > W ? hx - 14 - 190 : hx + 14) : 0;

  return (
    <div ref={boxRef} className={styles.box}>
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={title}
        className={styles.svg}
      >
        {TICKS.map((v) => (
          <g key={v}>
            <line x1={m.l} x2={m.l + pw} y1={Y(v)} y2={Y(v)} className={v === 10_000 ? styles.base : styles.grid} />
            <text x={m.l - 10} y={Y(v) + 4} className={v === 10_000 ? styles.axisStart : styles.axis} textAnchor="end">
              {TICK_LABEL[v]}
            </text>
          </g>
        ))}

        {years.map((y, i) => (
          <g key={y.yr}>
            {i > 0 && <line x1={X(y.s)} x2={X(y.s)} y1={m.t} y2={m.t + ph + 6} className={styles.grid} />}
            {X(y.e) - X(y.s) > 36 && (
              <text x={(X(y.s) + X(y.e)) / 2} y={m.t + ph + 20} className={styles.axis} textAnchor="middle">
                {y.yr}
              </text>
            )}
          </g>
        ))}

        <path d={paths.gap} className={styles.gap} />
        {hold && <path d={paths.hold} className={styles.hold} />}
        <path d={paths.no} className={styles.lineNo} />
        <path d={paths.at} className={styles.lineAt} />

        {ends.map((e) => (
          <text key={e.cls} x={m.l + pw + 10} y={e.y} className={e.cls}>
            {gbp(e.v)}
          </text>
        ))}

        {months.map((mo, i) => {
          const xa = X(Math.max(mo.start, x0));
          const xb = X(Math.min(mo.end, x1));
          return (
            <rect
              key={mo.label}
              x={xa + 0.75}
              y={stripY}
              width={Math.max(1, xb - xa - 1.5)}
              height={14}
              rx={2}
              className={monthNet[i] > 0 ? styles.mGain : styles.mLoss}
            >
              <title>{`${mo.label}: ${signed(monthNet[i], 0)} pips at ${costLabel}`}</title>
            </rect>
          );
        })}

        {hover !== null && (
          <g pointerEvents="none">
            <line x1={hx} x2={hx} y1={m.t} y2={m.t + ph} className={styles.cross} />
            <circle cx={hx} cy={Y(noCost[hover])} r={4.5} className={styles.dotNo} />
            <circle cx={hx} cy={Y(atCost[hover])} r={4.5} className={styles.dotAt} />
          </g>
        )}
        <rect
          x={m.l}
          y={m.t}
          width={pw}
          height={ph}
          fill="transparent"
          onPointerMove={onMove}
          onPointerDown={onMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover !== null && (
        <div className={styles.tip} style={{ left: tipLeft }}>
          <div className={styles.tipDate}>{niceDate(isoDates[hover])}</div>
          <div className={styles.tipNo}>No costs {gbp(noCost[hover])}</div>
          <div className={styles.tipAt}>At {costLabel} {gbp(atCost[hover])}</div>
        </div>
      )}
    </div>
  );
}
