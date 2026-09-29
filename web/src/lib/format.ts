export const MINUS = "−";

export function gbp(v: number): string {
  if (v >= 1000) return "£" + Math.round(v).toLocaleString("en-GB");
  if (v >= 10) return "£" + Math.round(v);
  return "£" + v.toFixed(2);
}

export function pct(v: number, digits = 1): string {
  return (v * 100).toFixed(digits) + "%";
}

export function signed(v: number, digits = 2): string {
  const s = v > 0 ? "+" : v < 0 ? MINUS : "";
  return s + Math.abs(v).toFixed(digits);
}

export function int(v: number): string {
  return Math.round(v).toLocaleString("en-GB");
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function niceDate(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

export function toMs(iso: string): number {
  return Date.parse(iso + "T00:00:00Z");
}
