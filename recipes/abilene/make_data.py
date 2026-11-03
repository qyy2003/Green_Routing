"""Download the Abilene archive (if needed), clean it, write the two forecasting mirrors.

usage: python make_data.py
writes $GR_ARTIFACTS_DIR/abilene/{mirror_links, mirror_total, dataset.pkl}
"""
import pickle

from green_routing.dataset import abilene
from green_routing.paths import ARTIFACTS_DIR, reference_dir

SRC = abilene.download(reference_dir() / "abilene")
OUT = ARTIFACTS_DIR / "abilene"
OUT.mkdir(parents=True, exist_ok=True)
ds = abilene.load(SRC)
S = abilene.series(ds)
print(f"{len(ds['X'])} five-minute steps {ds['X'].index.min()} .. {ds['X'].index.max()}, "
      f"{int(ds['X'].isna().all(axis=1).sum())} missing; total {S['total'].mean() / 1e3:.2f} Gbit/s mean")
print(len(abilene.write_mirror(ds, OUT / "mirror_links", "links")), "link and door series;",
      len(abilene.write_mirror(ds, OUT / "mirror_total", "total")), "total series")
pickle.dump(ds, open(OUT / "dataset.pkl", "wb"))
