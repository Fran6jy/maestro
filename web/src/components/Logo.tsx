/** MAESTRO mark: an "M" drawn as a price path; the last leg is the trade that has to pay. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#0e131a" stroke="#27303c" />
      <polyline
        points="6,23 11,9 16,18 21,11"
        fill="none"
        stroke="#edf1f5"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polyline
        points="21,11 26,23"
        fill="none"
        stroke="#ffb547"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function Logo() {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
      <LogoMark />
      <span
        style={{
          fontFamily: "var(--font-display)",
          fontWeight: 650,
          fontStretch: "85%",
          letterSpacing: "0.12em",
          fontSize: "0.95rem",
        }}
      >
        MAESTRO
      </span>
    </span>
  );
}
