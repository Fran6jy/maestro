"use client";

import { useEffect, useTransition } from "react";
import { useRouter } from "next/navigation";
import styles from "./live.module.css";

const CHECK_EVERY_MS = 5 * 60 * 1000;

export default function LiveRefresh() {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  const refresh = () => startTransition(() => router.refresh());

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, CHECK_EVERY_MS);
    return () => window.clearInterval(timer);
  });

  return (
    <button type="button" className={styles.refresh} onClick={refresh} disabled={pending}>
      {pending ? "Checking…" : "Check for update"}
    </button>
  );
}
