import os
import numpy as np
import pandas as pd


def report_top_tiles(e_tile, f_tile, t_tile, random_time, geocent_time, event, det,
                     y_cut=100.0, gap=1.0, label="", outdir="."):
    """y>y_cut の巨大タイルを時刻でまとめて「イベント」として表示し、CSVに追記する。
    e_tile, f_tile, t_tile : run_test_band_qavg が返す間引き前の全タイル配列。
    y = e_tile*ln2 (run_test_band_qavg と同じ規格化)。ガウスなら y>100 は e^-100≈0 個。
    gap [s] 以内に連続するタイルは1イベントにまとめる。seg# は random_time の何番目の窓か。
    戻り値: イベントごとに1行の DataFrame (巨大タイルなしなら空のDataFrame)。"""
    y_all = np.asarray(e_tile) * np.log(2.0)
    t_all = np.asarray(t_tile)
    f_all = np.asarray(f_tile)
    cols = ["band", "big_tile_time", "seg", "freq", "tile_power", "tile_count"]

    sel = np.where(y_all > y_cut)[0]
    if sel.size == 0:
        print(f"{label}: y>{y_cut:g} のタイルなし")
        return pd.DataFrame(columns=cols)

    sel = sel[np.argsort(t_all[sel])]
    ev = np.split(sel, np.where(np.diff(t_all[sel]) > gap)[0] + 1)
    rt = np.asarray(random_time)
    print(f"{label}: y>{y_cut:g} のタイル {sel.size} 個 → {len(ev)} イベント")

    rows = []
    for g in ev:
        k = g[np.argmax(y_all[g])]
        seg = int(np.searchsorted(rt, t_all[k], side="right") - 1)
        dt = t_all[k] - geocent_time
        print(f"  t-t_merger={dt:+9.1f} s  seg#{seg:3d}  "
              f"f={f_all[k]:6.1f} Hz  ymax={y_all[k]:9.1f}  tiles={len(g)}")
        rows.append({"band": label, "big_tile_time": dt, "seg": seg,
                     "freq": f_all[k], "tile_power": y_all[k], "tile_count": len(g)})

    df = pd.DataFrame(rows, columns=cols)       # ← ループの外で1回だけ作る
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{event}{det}_big_tiles_time.csv")
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    return df


if __name__ == "__main__":      # 合成データで動作確認
    import tempfile
    rng = np.random.default_rng(0)
    gc = 1000.0
    rt = gc - 2048 + 20.4 * np.arange(100)
    t = np.concatenate([r + rng.uniform(0, 2, 300) for r in rt])
    f = rng.uniform(30, 40, t.size)
    e = rng.exponential(1.0, t.size) / np.log(2)
    for s_i, amp in [(10, 2e4), (11, 8e3), (57, 3e4)]:
        idx = np.where((t >= rt[s_i]) & (t < rt[s_i] + 2))[0][:3]
        e[idx] = amp / np.log(2)
        t[idx] = rt[s_i] + 0.8 + 0.01 * np.arange(len(idx))
    with tempfile.TemporaryDirectory() as d:
        out = report_top_tiles(e, f, t, rt, gc, "TEST", "L1",
                               label="synthetic 30-40Hz", outdir=d)
        print(out)
        print(open(os.path.join(d, "TESTL1_big_tiles_time.csv")).read())