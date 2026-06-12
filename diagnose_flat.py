"""Diagnose why the system never trades — trace the signal → risk → decision chain."""
import pandas as pd, numpy as np

d = pd.read_parquet(r"C:\tmp\maestro_outputs\EUR_USD\decisions_split_000.parquet")
print("=== RISK DECISION breakdown (split 000) ===")
print("action value counts:")
print(d["action"].value_counts())
print("\nrisk_reason value counts:")
print(d["risk_reason"].value_counts())
print("\nunits stats: min=%.0f max=%.0f nonzero=%d" % (d["units"].min(), d["units"].max(), (d["units"] != 0).sum()))
print("position_fraction: min=%.4f max=%.4f mean=%.4f" % (
    d["position_fraction"].min(), d["position_fraction"].max(), d["position_fraction"].mean()))
print("compliant: %d / %d" % (d["compliant"].sum(), len(d)))
print("equity non-null: %d / %d" % (d["equity"].notna().sum(), len(d)))
