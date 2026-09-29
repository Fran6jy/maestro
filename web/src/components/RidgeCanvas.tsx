"use client";

import { useEffect, useRef } from "react";
import styles from "./Hero.module.css";

interface Props {
  paths: number[][];
  days: string[];
}

const VISIBLE = 44;       // ridges on screen at once
const SPEED = 0.32;       // ridges per second flowing toward the viewer
const BG = "#07090c";

/**
 * Real EUR/USD sessions as a ridgeline landscape. Each ridge is one trading day,
 * midnight to midnight UTC, drawn as pips from that day's first price.
 * Amber ridges mark every eleventh day; the cursor reads the time of day.
 */
export default function RidgeCanvas({ paths, days }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const readoutRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const readout = readoutRef.current;
    if (!canvas || !readout || paths.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const N = paths.length;
    const P = paths[0].length;
    const abs = paths.flat().map(Math.abs).sort((a, b) => a - b);
    const scale = abs[Math.floor(abs.length * 0.96)] || 40;

    let W = 0;
    let H = 0;
    let raf = 0;
    let visible = true;
    const t0 = performance.now();
    const mouse = { x: 0.6, y: 0.5, tx: 0.6, ty: 0.5, active: false };

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      W = r.width;
      H = r.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now: number) => {
      mouse.x += (mouse.tx - mouse.x) * 0.06;
      mouse.y += (mouse.ty - mouse.y) * 0.06;
      const t = reduce ? 0 : (now - t0) / 1000;
      const shift = t * SPEED;
      const base = Math.floor(shift);
      const frac = shift - base;

      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, W, H);

      const horizon = H * 0.34 + (mouse.y - 0.5) * H * 0.05;
      const floor = H * 1.04;
      const narrow = W < 700;

      let frontIdx = 0;
      for (let k = 0; k < VISIBLE; k++) {
        const d = (k + frac) / VISIBLE;                       // 0 = far, 1 = near
        const idx = (((base + VISIBLE - k) % N) + N) % N;
        const path = paths[idx];
        const yb = horizon + (floor - horizon) * Math.pow(d, 1.55);
        const amp = (narrow ? 10 : 12) + d * H * (narrow ? 0.16 : 0.2);
        const width = W * (0.62 + d * 0.78);
        const cx = W * 0.5 + (mouse.x - 0.5) * W * 0.1 * (1 - d);
        const x0 = cx - width / 2;
        const step = width / (P - 1);

        ctx.beginPath();
        for (let i = 0; i < P; i++) {
          const x = x0 + i * step;
          const y = yb - (path[i] / scale) * amp;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        // occlude the ridges behind
        ctx.save();
        ctx.lineTo(x0 + width, floor + 40);
        ctx.lineTo(x0, floor + 40);
        ctx.closePath();
        ctx.fillStyle = BG;
        ctx.fill();
        ctx.restore();

        ctx.beginPath();
        for (let i = 0; i < P; i++) {
          const x = x0 + i * step;
          const y = yb - (path[i] / scale) * amp;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        const fadeIn = Math.min(1, d / 0.12);
        const fadeOut = Math.min(1, (1 - d) / 0.1);
        const a = (0.1 + 0.72 * Math.pow(d, 1.25)) * fadeIn * fadeOut;
        const amber = idx % 11 === 0;
        ctx.lineWidth = 0.8 + d * 1.2;
        if (amber) {
          ctx.strokeStyle = `rgba(255, 181, 71, ${Math.min(1, a * 1.35)})`;
          ctx.shadowColor = "rgba(255, 170, 60, 0.55)";
          ctx.shadowBlur = 12 * d;
        } else {
          ctx.strokeStyle = `rgba(163, 173, 186, ${a})`;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
        if (k === VISIBLE - 3) frontIdx = idx;
      }

      // cursor reads time of day on the near ridge
      if (mouse.active) {
        const d = 1 - 3 / VISIBLE;
        const width = W * (0.62 + d * 0.78);
        const x0 = W * 0.5 - width / 2;
        const f = Math.min(1, Math.max(0, (mouse.tx * W - x0) / width));
        const minutes = Math.round((f * 24 * 60) / 5) * 5;
        const hh = String(Math.floor(minutes / 60) % 24).padStart(2, "0");
        const mm = String(minutes % 60).padStart(2, "0");
        const i = Math.round(f * (P - 1));
        const pips = paths[frontIdx][i];
        readout.style.transform = `translate(${mouse.tx * W}px, ${Math.max(horizon + 24, mouse.ty * H - 56)}px)`;
        readout.textContent = `${days[frontIdx]} · ${hh}:${mm} UTC · ${pips > 0 ? "+" : pips < 0 ? "−" : ""}${Math.abs(pips)} pips`;
        ctx.strokeStyle = "rgba(255, 181, 71, 0.28)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mouse.tx * W, horizon - 20);
        ctx.lineTo(mouse.tx * W, H);
        ctx.stroke();
      }

      if (!reduce && visible) raf = requestAnimationFrame(draw);
    };

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      mouse.tx = (e.clientX - r.left) / r.width;
      mouse.ty = (e.clientY - r.top) / r.height;
      mouse.active = e.pointerType === "mouse" && mouse.ty > 0.3;
      readout.hidden = !mouse.active;
      if (reduce) draw(performance.now());
    };
    const onLeave = () => {
      mouse.active = false;
      readout.hidden = true;
      mouse.tx = 0.6;
      mouse.ty = 0.5;
    };

    const io = new IntersectionObserver(([entry]) => {
      const was = visible;
      visible = entry.isIntersecting;
      if (visible && !was && !reduce) raf = requestAnimationFrame(draw);
    });

    resize();
    draw(performance.now());
    const ro = new ResizeObserver(() => {
      resize();
      if (reduce) draw(performance.now());
    });
    ro.observe(canvas);
    io.observe(canvas);
    const host = canvas.parentElement ?? canvas;
    host.addEventListener("pointermove", onMove);
    host.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      host.removeEventListener("pointermove", onMove);
      host.removeEventListener("pointerleave", onLeave);
    };
  }, [paths, days]);

  return (
    <>
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
      <div ref={readoutRef} className={styles.readout} hidden aria-hidden="true" />
    </>
  );
}
