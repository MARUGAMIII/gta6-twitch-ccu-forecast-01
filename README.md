# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/marugamiii/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/marugamiii/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (19 Nov 2026, UTC), after the first hour — five
quantiles (5%, 25%, 50%, 75%, 95%) per hour, hours 2–24.

## How it works

One file, `main.py` (Python 3.13, standard library only, no network,
deterministic). The case gives a single anchor — hour 1's print — and the
program fits a per-hour log-log regression on the case's own comparables
(past big game launches' hourly Twitch category averages):

    log viewers(hour h) = a_h + b_h * log viewers(hour 1)

fitted only on launches whose shape matches GTA VI's: a rolling
midnight-local console unlock, single-player, no early access, no
simultaneous global unlock. The split comes from each comparable's unlock
notes plus a data check (peak / hour 1 above 20 means hour 1 was recorded
before the game was playable). The fitted slope is about 1.0 just after
midnight and falls to about 0.55 for the evening peak — the peak scales
like viewers(hour 1) ** 0.55 — because the launch-day curve is the unlock
wave rolling through time zones on top of Twitch's diurnal cycle. The
spread is the honest one: the inter-quartile range of the regression's
residuals as a normal sigma, floored at 0.08 and widened by 1.1.

Validated on the four practice cases against the true SullyGnome
launch-day counts, with the contest's pinball loss on the log scale:
**mean loss 0.169**, against 0.387 for the house baseline — a ~56%
improvement — with a typical median miss of ~29% and 90% coverage of
~0.87. If the regression cannot be fitted, the program falls back to the
house baseline so the output is always a valid forecast.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with the case at `/input`
and writes the forecast to `/output`. One run takes about 0.15 s per
case, and two runs on the same case write byte-identical outputs.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
