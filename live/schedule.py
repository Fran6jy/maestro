"""UTC publication times for the live trial's Foundation snapshots."""

import pandas as pd

PUBLISH_SLOTS = {
    "dawn": pd.Timedelta(minutes=20),
    "day": pd.Timedelta(hours=12, minutes=20),
    "dusk": pd.Timedelta(hours=20, minutes=20),
}
