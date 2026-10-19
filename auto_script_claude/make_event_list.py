#!/usr/bin/env python
"""GWOSC からイベント一覧と合体時刻を取得し、L1 の 4096 s / 4 kHz HDF5 をダウンロードする。
出力: events.csv (name, gps, det, file, catalog)

使い方:
  python make_event_list.py                     # O3 (GWTC-2.1, GWTC-3) の confident イベント
  python make_event_list.py --max 5             # 試しに5イベントだけ
  python make_event_list.py --no-download       # 一覧だけ作る
"""
import argparse
import csv
import os
import re
import urllib.request

from gwosc.datasets import find_datasets, event_gps, event_detectors
from gwosc.locate import get_event_urls

CATALOGS = ["GWTC-2.1-confident", "GWTC-3-confident"]   # O3a, O3b以降
#CATALOGS = ["GWTC-2.1-confident", "GWTC-3-confident","GWTC-4.1","GWTC-5.0"]
DET = "L1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", nargs="+", default=CATALOGS)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--outdir", default="hdf5")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--out", default="events.csv")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    seen = {}
    for cat in a.catalogs:
        for full in find_datasets(type="events", catalog=cat):
            name = re.sub(r"-v\d+$", "", full)          # "GW200129_065458-v1" -> "GW200129_065458"
            seen[name] = (full, cat)                     # 後のカタログで上書き (最新版を使う)
    names = sorted(seen)
    print(f"{len(names)} イベント ({', '.join(a.catalogs)})")

    rows = []
    for name in names:
        if a.max and len(rows) >= a.max:
            break
        full, cat = seen[name]
        try:
            if DET not in event_detectors(full, catalog=cat):
                print(f"  skip {name}: {DET} なし"); continue
            gps = float(event_gps(full, catalog=cat))
            if gps < 1238166018:                       # O3 より前 (O1/O2) は除く
                print(f"  skip {name}: O1/O2"); continue
            urls = [u for u in get_event_urls(full, catalog=cat, detector=DET,
                                              sample_rate=4096, format="hdf5")
                    if u.endswith("-4096.hdf5")]
            if not urls:
                print(f"  skip {name}: 4096 s ファイルなし"); continue
            fn = os.path.join(a.outdir, os.path.basename(urls[0]))
            if not a.no_download and not os.path.exists(fn):
                print(f"  download {name} -> {fn}")
                urllib.request.urlretrieve(urls[0], fn + ".part")
                os.replace(fn + ".part", fn)
            rows.append(dict(name=name, gps=f"{gps:.3f}", det=DET, file=fn, catalog=cat))
            print(f"  ok {name}  gps={gps:.3f}")
        except Exception as e:
            print(f"  skip {name}: {e}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "gps", "det", "file", "catalog"])
        w.writeheader(); w.writerows(rows)
    print(f"-> {a.out}: {len(rows)} イベント")


if __name__ == "__main__":
    main()
