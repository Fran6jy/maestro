# MAESTRO website

The public research site at **https://maestro-research.vercel.app**, built with Next.js 16.

It never computes results itself: every figure comes from `src/data/*.json`, exported from the tested
research pipeline by `python -m maestro.demo.export_web_data` (run from the folder above the repo).
The Strategy Lab re-prices trades at any cost exactly, because trading cost is linear in the number
of trades.

```bash
npm install
npm run dev              # http://localhost:3000
npm run build            # production build, all pages static
vercel deploy --prod     # publish
```

Read `AGENTS.md` before changing the code: Next.js 16 has breaking changes from earlier versions.

| Path | What it is |
|---|---|
| `src/app/page.tsx` | Home page composition |
| `src/app/method/` | Methodology page |
| `src/app/opengraph-image.tsx` | Share card, drawn from real EUR/USD sessions |
| `src/components/StrategyLab.tsx` | Strategy picker, cost slider, leaderboard |
| `src/components/RidgeCanvas.tsx` | Hero ridgeline animation |
| `src/lib/metrics.ts` | Cost re-pricing, equity, Sharpe |
| `src/data/` | Exported results (do not edit by hand) |
