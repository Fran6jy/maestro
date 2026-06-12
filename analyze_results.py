"""Quick analysis of completed WFA splits — reads decision parquets, computes metrics."""
import pandas as pd, numpy as np, glob, os

ANNUAL = np.sqrt(252 * 78)
feat = pd.read_parquet(r"C:\tmp\maestro_data\EUR_USD_features.parquet")
ret_col = "log_return_1" if "log_return_1" in feat.columns else None

files = sorted(glob.glob(r"C:\tmp\maestro_outputs\EUR_USD\decisions_split_*.parquet"))
print(f"Found {len(files)} completed splits\n")
print(f"{'Split':<7}{'Bars':>7}{'Trades':>8}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>8}{'Hit%':>7}{'CumRet%':>9}")
print("-" * 64)

all_net = []
total_trades = 0
for f in files:
    d = pd.read_parquet(f)
    sid = os.path.basename(f).split("_")[-1].split(".")[0]
    units = d["units"].fillna(0)
    direction = np.sign(units.values)
    active = direction != 0
    ntr = int(active.sum())
    total_trades += ntr

    if d["equity"].notna().any():
        eq = d["equity"].ffill()
        net = eq.pct_change().fillna(0)
    else:
        net = d["net_return"].fillna(0)

    na = net[net != 0]
    sharpe = float(na.mean() / (na.std() + 1e-10) * ANNUAL) if len(na) > 10 else 0.0
    dn = na[na < 0].std() + 1e-10
    sortino = float(na.mean() / dn * ANNUAL) if len(na) > 10 else 0.0
    cum = (1 + net).cumprod()
    maxdd = float(((cum - cum.cummax()) / cum.cummax()).min()) * -1
    cumret = float(cum.iloc[-1] - 1)

    if ret_col:
        fwd = feat[ret_col].reindex(d.index).fillna(0).values
        hit = float((np.sign(fwd[active]) == direction[active]).mean()) if ntr > 0 else 0.0
    else:
        hit = float("nan")

    print(f"{sid:<7}{len(d):>7}{ntr:>8}{sharpe:>9.3f}{sortino:>9.3f}{maxdd*100:>7.1f}%{hit*100:>6.1f}%{cumret*100:>8.2f}%")
    all_net.append(net)

agg = pd.concat(all_net)
aa = agg[agg != 0]
print("-" * 64)
print(f"AGGREGATE across {len(files)} splits:")
sharpe_agg = aa.mean() / (aa.std() + 1e-10) * ANNUAL if len(aa) > 10 else 0.0
print(f"  Net Sharpe:    {sharpe_agg:>7.3f}    (target >1.5 | MSc baseline 0.599)")
cumA = (1 + agg).cumprod()
print(f"  Max Drawdown:  {((cumA - cumA.cummax()) / cumA.cummax()).min() * -100:>6.1f}%    (target <15%)")
print(f"  Total Trades:  {total_trades}")
print(f"  Active bars:   {len(aa)} / {len(agg)} ({100*len(aa)/len(agg):.1f}%)")
