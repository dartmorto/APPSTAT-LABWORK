##в датасете нет готовых признаков, поэтому их нужно вычислить из карт деффектов (схема пластинки, где отмечены дефектные кристаллы (визуализация есть в convert_maps.py))
## мы выбрали parquet, потому что он быстрее и компактнее csv (у меня закончилась память)
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R_CENTER = 0.5   # центр: r < 0.5
R_EDGE = 0.8     # край:  r > 0.8


def wafer_features(m):
    """Признаки карты: число кристаллов, дефекты и радиус центроида."""
    ys, xs = np.nonzero(m > 0)  # все кристаллы на пластине
    if len(ys) == 0:
        return 0, 0, 0, 0, np.nan
    cy = (m.shape[0] - 1) / 2
    cx = (m.shape[1] - 1) / 2
    r_raw = np.hypot(ys - cy, xs - cx)
    r_max = r_raw.max()
    r = r_raw / r_max                                # нормированный радиус
    bad = m[ys, xs] == 2                             # дефектные кристаллы
    n_defect = int(bad.sum())
    n_center = int((bad & (r < R_CENTER)).sum())
    n_edge = int((bad & (r > R_EDGE)).sum())
    if n_defect > 0:
        r_def = float(np.hypot(ys[bad].mean() - cy, xs[bad].mean() - cx) / r_max)
    else:
        r_def = np.nan
    return len(ys), n_defect, n_center, n_edge, r_def


def main():
    df = pd.read_csv(ROOT / "LSWMD_meta.csv")
    print(df["failureType"].value_counts(dropna=False).head())

    # однородная выборка: класс none и самый частый dieSize
    none = df[df["failureType"] == "none"]
    die_size = none["dieSize"].value_counts().index[0]
    sub = none[none["dieSize"] == die_size].copy()
    sub["map_idx"] = sub.index
    print(f"dieSize = {die_size}, пластин: {len(sub)}")

    with open(ROOT / "LSWMD_wafermaps.pkl", "rb") as f:
        maps = pickle.load(f)

    feats = pd.DataFrame(
        [wafer_features(maps[i]) for i in sub["map_idx"]],
        columns=["n_die", "n_defect", "n_center", "n_edge", "r_def"],
        index=sub.index,
    )
    sub = pd.concat([sub, feats], axis=1).reset_index(drop=True)
    sub.to_parquet(ROOT / "wm811k_features.parquet", index=False)


if __name__ == "__main__":
    main()
