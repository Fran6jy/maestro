"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import Logo from "./Logo";
import styles from "./Nav.module.css";

const LINKS = [
  { href: "/#lab", label: "Strategy Lab" },
  { href: "/#msc", label: "The MSc" },
  { href: "/#agents", label: "The agents" },
  { href: "/#roadmap", label: "Roadmap" },
  { href: "/method", label: "Method" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const current = (href: string) => (href === pathname ? "page" : undefined);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={styles.wrap}>
      <nav className={`${styles.bar} ${scrolled ? styles.solid : ""}`} aria-label="Main">
        <Link href="/" className={styles.brand} aria-label="MAESTRO home" onClick={() => setOpen(false)}>
          <Logo />
        </Link>
        <ul className={styles.links}>
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={current(l.href)}>{l.label}</Link>
            </li>
          ))}
        </ul>
        <button
          type="button"
          className={styles.menuBtn}
          aria-expanded={open}
          aria-controls="mobile-menu"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="visually-hidden">{open ? "Close menu" : "Open menu"}</span>
          <span className={`${styles.burger} ${open ? styles.burgerOpen : ""}`} aria-hidden="true" />
        </button>
      </nav>
      <div id="mobile-menu" className={styles.sheet} hidden={!open}>
        <ul>
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} onClick={() => setOpen(false)} aria-current={current(l.href)}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </header>
  );
}
