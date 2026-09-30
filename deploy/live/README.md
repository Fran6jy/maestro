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
or an order. (Still to come: a daily reconciliation of the journal against the backtest.)
