# orb-gooey-trader

Paper-trading **signal bot** for the "ORB + Gooey Model" reversal strategy on **/NQ**, using:

- 15-minute Opening Range (9:30–9:45 ET)
- Liquidity sweeps beyond ORB
- SMT divergence confirmation versus **/ES**
- FVG formation + iFVG inversion confirmation (1m/3m/5m)

No live orders are placed. The bot logs signals and can send alerts to Campfire.

## Setup

```bash
cd ~/projects/orb-gooey-trader
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config/secrets.yaml.example config/secrets.yaml
# edit config/secrets.yaml with Schwab + Campfire credentials
```

## Run

```bash
python -m src.main
```

## Notes

- Schwab authentication: first run may require an interactive browser OAuth flow.
- Candle data: the bot fetches 1m data and resamples to 3m/5m/15m/4h.

