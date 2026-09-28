"""Create a reproducible SYNTHETIC teaching dataset. Not real survey responses."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
n = 1500
screen = np.round(rng.uniform(1, 12, n), 1)
social = np.round(screen * rng.uniform(.05, .65, n), 1)
gaming = np.round((screen-social) * rng.uniform(0, .5, n), 1)
sleep = np.round(np.clip(8.5 - .18*screen + rng.normal(0, .9, n), 4, 10), 1)
study = np.round(np.clip(rng.normal(4, 1.4, n) - .09*screen, .5, 9), 1)
checks = np.clip(np.rint(18 + screen*11 + rng.normal(0, 24, n)), 5, 240).astype(int)
night = np.clip(np.rint(screen*8 + rng.normal(0, 24, n)), 0, 180).astype(int)
# Target is generated from an artificial formula plus noise. This is ONLY a software demo.
latent = .7*screen + .55*social + .35*gaming + .018*checks + .016*night - .45*sleep - .22*study + rng.normal(0, 2.1, n)
q1, q2 = np.quantile(latent, [.33, .67])
risk = np.where(latent<q1,'Low',np.where(latent<q2,'Moderate','High'))
df = pd.DataFrame({'daily_screen_time_hours':screen,'social_media_hours':social,'gaming_hours':gaming,'phone_checks_per_day':checks,'night_use_minutes':night,'sleep_hours':sleep,'study_hours':study,'risk_level':risk})
path=ROOT/'data'/'smartphone_usage_synthetic.csv';path.parent.mkdir(exist_ok=True)
df.to_csv(path,index=False)
print(f'Wrote {len(df)} SYNTHETIC rows to {path}')
print(df.risk_level.value_counts().to_string())
