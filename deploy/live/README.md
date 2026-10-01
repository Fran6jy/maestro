# MAESTRO live trial

Paper-trades MAESTRO and the baselines on every 5-minute EUR/USD bar, and sends one
MAESTRO variant's positions to an **OANDA practice account**. It cannot reach a
real-money account: `live/oanda.py` refuses any server but OANDA's practice API.

## On the VM (once)

```bash
mkdir -p ~/maestro-live/state/models && cd ~/maestro-live
# copy Dockerfile and docker-compose.yml from this folder, then create .env yourself:
#   OANDA_API_KEY=...        (practice account token)
#   OANDA_ACCOUNT_ID=...     (practice account, 101-...)
#   FRED_API_KEY=...
chmod 600 .env
```

## Deploy a model (from the laptop, which has the GPU)

```bash
python -m maestro.live.deploy --out C:\tmp\maestro_live\models\<date>
scp -r C:\tmp\maestro_live\models\<date> <vm>:~/maestro-live/state/models/<date>
ssh <vm> 'cd ~/maestro-live/state/models && ln -sfn <date> current'
```

## Run

```bash
MAESTRO_COMMIT=<pushed commit> docker compose up -d --build
docker compose logs -f              # or: cat state/status.json
docker compose down                 # stop
```

The first start fetches 13 months of candles from OANDA and FRED's history, then
handles each bar about 10 seconds after it closes. A restart never repeats a bar
or an order.

## Check on it

```bash
docker exec maestro-live python -m maestro.live.report     # health, every strategy in pips, orders
```

## Nightly snapshot

`maestro-publish` builds a snapshot from the journal at 00:20 UTC (`live/snapshot.py`:
daily returns, pips, trades, practice fills against paper, bars recorded against bars
the market was open; never prices) and pushes it to the `Fran6jy/maestro-live`
repository as `snapshot.json` plus `history/<date>.json`. The website's live page
reads `snapshot.json`; a GitHub Action in that repository runs `live/check.py` after
every snapshot and every night, and fails (so GitHub emails the owner) if the
snapshot is stale, the loop stopped or errored, or bars were missed.

Once:

1. Create the `maestro-live` repository (private during the shakedown) and copy in
   the files from `snapshot-repo/` here.
2. On the VM, make a key that only this service uses:
   `ssh-keygen -t ed25519 -N "" -C maestro-publish -f ~/maestro-live/keys/maestro_live`
3. Add `keys/maestro_live.pub` to the repository as a deploy key with write access.
4. `docker compose up -d --build` starts both services.

## Going public (when the trial proper starts)

1. Regenerate the expectations from the frozen design, commit and push them *before*
   the trial starts: `python -m maestro.live.expectations`.
2. Deploy the trial's model and restart with `--orders <variant>` and
   `MAESTRO_LIVE_PHASE=trial`.
3. Set `LIVE_INDEX=1` on Vercel (the page has been public but unlisted since the
   shakedown: `maestro-live` public, `LIVE_PUBLIC=1`), add the Live link to the site's
   navigation and redeploy.
