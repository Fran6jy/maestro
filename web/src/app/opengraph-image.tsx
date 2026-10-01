import { ImageResponse } from "next/og";
import { RESULTS, RIDGE, STRATEGY } from "@/lib/data";
import { finalBalance } from "@/lib/metrics";
import { gbp } from "@/lib/format";

export const alt = "MAESTRO: the cost of being right. £10,000 on a simple trading model, before and after costs.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  const lr = STRATEGY.logreg_lag5;
  const before = gbp(finalBalance(lr, 0));
  const after = gbp(finalBalance(lr, RESULTS.refCostPips));

  // A handful of real EUR/USD sessions as ridges along the bottom of the card.
  const rows = 16;
  const pick = RIDGE.paths.filter((_, i) => i % Math.floor(RIDGE.paths.length / rows) === 0).slice(0, rows);
  const W = 1200;
  const ridges = pick.map((p, k) => {
    const d = (k + 1) / rows;
    const yb = 440 + Math.pow(d, 1.5) * 220;
    const amp = 6 + d * 48;
    const width = W * (0.7 + d * 0.6);
    const x0 = W / 2 - width / 2;
    const pts = p.map((v, i) => `${(x0 + (i / (p.length - 1)) * width).toFixed(1)},${(yb - (v / 45) * amp).toFixed(1)}`).join(" ");
    const amber = k % 5 === 3;
    return { pts, stroke: amber ? `rgba(255,181,71,${0.25 + d * 0.7})` : `rgba(163,173,186,${0.08 + d * 0.5})` };
  });

  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: "#07090c", padding: "64px 72px", position: "relative", color: "#edf1f5" }}>
        <svg width={1200} height={630} viewBox="0 0 1200 630" style={{ position: "absolute", left: 0, top: 0 }}>
          {ridges.map((r, i) => (
            <polyline key={i} points={r.pts} fill="none" stroke={r.stroke} strokeWidth={1.6} />
          ))}
        </svg>
        <div style={{ display: "flex", alignItems: "center", fontSize: 26, color: "#a3adba" }}>
          MAESTRO · PhD research on EUR/USD trading
        </div>
        <div style={{ display: "flex", flexDirection: "column", marginTop: 36, fontSize: 96, fontWeight: 700, lineHeight: 0.95, letterSpacing: -3 }}>
          <span>The cost of</span>
          <span style={{ color: "#ffb547" }}>being right.</span>
        </div>
        <div style={{ display: "flex", marginTop: 34, fontSize: 30, color: "#a3adba" }}>
          £10,000 on a simple model:&nbsp;<span style={{ color: "#ffb547" }}>{before}</span>&nbsp;before costs,&nbsp;
          <span style={{ color: "#8db4ff" }}>{after}</span>&nbsp;after.
        </div>
      </div>
    ),
    size,
  );
}
