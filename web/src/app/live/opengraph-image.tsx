import { ImageResponse } from "next/og";
import { EXPECT, balanceChange, balancePath, getSnapshot } from "@/lib/live";

export const alt = "MAESTRO live trial: four strategies trading EUR/USD with £10,000 of pretend money each.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const revalidate = 3600;

const FEATURED = ["risk_cost_filter", "bollinger_20_2", "logreg_lag5", "buy_hold"];
const MINUS = "−";

function money(v: number, signed: boolean): string {
  const a = Math.abs(v);
  const s = a >= 100 ? Math.round(a).toLocaleString("en-GB") : a.toFixed(2);
  if (!signed) return `£${s}`;
  return `${v > 0.004 ? "+" : v < -0.004 ? MINUS : ""}£${s}`;
}

export default async function Image() {
  const snap = await getSnapshot();
  const ready = !!snap && !!snap.start && snap.days.length > 0;
  const start = snap?.start_gbp ?? EXPECT.start_balance;
  const maestro = snap?.order_strategy ?? FEATURED[0];
  const keys = ready ? [maestro, ...FEATURED.slice(1)].filter((k) => snap!.strategies[k]) : [];
  const n = snap?.days.length ?? 0;

  // MAESTRO's predicted range and live line, faint, across the bottom of the card.
  const exp = EXPECT.strategies[maestro];
  const live = ready ? balancePath(snap!.strategies[maestro]?.daily.ref ?? [], start) : [];
  const span = Math.min(EXPECT.horizon_days, Math.max(10, live.length + 5));
  const ranges = exp ? exp.ranges.slice(0, span) : [];
  const lo = Math.min(0, ...ranges.map((q) => q[0]), ...live);
  const hi = Math.max(0, ...ranges.map((q) => q[4]), ...live);
  const y = (v: number) => 470 + ((hi - v) / (hi - lo || 1)) * 140;
  const x = (d: number) => 640 + (d / span) * 520;
  const band = (a: number, b: number) =>
    [`${x(0)},${y(0)}`, ...ranges.map((q, i) => `${x(i + 1)},${y(q[b])}`),
     ...ranges.map((q, i) => `${x(i + 1)},${y(q[a])}`).reverse()].join(" ");
  const path = [`${x(0)},${y(0)}`, ...live.map((v, i) => `${x(i + 1)},${y(v)}`)].join(" ");

  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: "#07090c",
        padding: "56px 64px", position: "relative", color: "#edf1f5" }}>
        {ranges.length > 0 && (
          <svg width={1200} height={630} viewBox="0 0 1200 630" style={{ position: "absolute", left: 0, top: 0 }}>
            <polygon points={band(0, 4)} fill="rgba(141,180,255,0.10)" />
            <polygon points={band(1, 3)} fill="rgba(141,180,255,0.16)" />
            {live.length > 0 && <polyline points={path} fill="none" stroke="#ffb547" strokeWidth={4} />}
          </svg>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 22, letterSpacing: 4, color: "#a3adba" }}>
          <div style={{ width: 12, height: 12, borderRadius: 6, background: "#ffb547" }} />
          {ready ? `MAESTRO · LIVE TRIAL · DAY ${n}` : "MAESTRO · LIVE TRIAL"}
          {snap?.phase === "shakedown" && (
            <div style={{ display: "flex", marginLeft: 8, padding: "4px 14px", borderRadius: 999, fontSize: 18,
              color: "#ffb547", background: "rgba(255,181,71,0.14)" }}>SHAKEDOWN</div>
          )}
        </div>
        <div style={{ display: "flex", marginTop: 26, fontSize: 64, fontWeight: 700, letterSpacing: -2, lineHeight: 1 }}>
          {ready ? "Where the pretend money stands" : "Starting soon"}
        </div>
        {ready ? (
          <div style={{ display: "flex", gap: 22, marginTop: 44 }}>
            {keys.map((k) => {
              const change = balanceChange(snap!.strategies[k].daily.ref, start);
              return (
                <div key={k} style={{ display: "flex", flexDirection: "column", width: 248, padding: "20px 22px",
                  borderRadius: 16, border: "1px solid #1a212b", background: "#0e131a" }}>
                  <div style={{ display: "flex", fontSize: 20, color: "#a3adba" }}>{EXPECT.strategies[k]?.label ?? k}</div>
                  <div style={{ display: "flex", fontSize: 46, fontWeight: 700, marginTop: 8 }}>{money(start + change, false)}</div>
                  <div style={{ display: "flex", fontSize: 22, marginTop: 4, color: Math.abs(change) < 0.005 ? "#a3adba" : change > 0 ? "#39d98a" : "#ff6b5b" }}>
                    {money(change, true)}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ display: "flex", marginTop: 30, fontSize: 30, color: "#a3adba" }}>
            Every strategy, £10,000 of pretend money, live prices, judged against a forecast written first.
          </div>
        )}
        <div style={{ display: "flex", marginTop: "auto", fontSize: 22, color: "#a3adba" }}>
          £10,000 of pretend money each · real EUR/USD prices · research, not advice
        </div>
      </div>
    ),
    size,
  );
}
