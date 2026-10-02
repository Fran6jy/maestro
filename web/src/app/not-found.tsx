import type { Metadata } from "next";
import Link from "next/link";
import styles from "./not-found.module.css";

export const metadata: Metadata = {
  title: "Not found",
  robots: { index: false, follow: false },
};

export default function NotFound() {
  return (
    <div className={`shell ${styles.page}`}>
      <p className={styles.code}>404</p>
      <h1>Nothing trades here.</h1>
      <p className="lede">
        The page you asked for doesn&rsquo;t exist, or has moved. Everything that does exist is one
        step away.
      </p>
      <div className={styles.actions}>
        <Link href="/#lab" className="btn btn-primary">
          Open the Strategy Lab <span className="arrow" aria-hidden="true">→</span>
        </Link>
        <Link href="/method" className="btn btn-ghost">
          How it was tested
        </Link>
      </div>
    </div>
  );
}
