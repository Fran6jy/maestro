"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Animates a positive money value toward its target on a log curve, so a jump
 * from £19,697 to £188 moves evenly instead of flashing through thousands.
 */
export function useMoneyTween(target: number, ms = 420): number {
  const [value, setValue] = useState(target);
  const current = useRef(target);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const from = Math.max(0.01, current.current);
    const to = Math.max(0.01, target);
    if (reduce || from === to) {
      current.current = to;
      setValue(to);
      return;
    }
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      const v = from * Math.pow(to / from, e);
      current.current = v;
      setValue(v);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);

  return value;
}
