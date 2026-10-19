#!/bin/bash
# 全自動: イベント一覧 → 背景テスト → PE キャンペーン → 集計
# 使い方: bash run_all.sh 40 10   (1イベントあたりの窓数, 同時実行数)
set -u
NWIN=${1:-40}; NPAR=${2:-10}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python make_event_list.py
EVS=$(tail -n +2 events.csv | cut -d, -f1)
# 背景テスト (1イベント1プロセス)
echo "$EVS" | xargs -P "$NPAR" -I{} sh -c 'python bg_auto.py --event {} > logs_bg_{}.txt 2>&1'
# PE: (イベント, 窓, ノイズ) を全部並べて NPAR 本ずつ。済み (summary.json がある) は飛ばす
for ev in $EVS; do for w in $(seq 0 $((NWIN-1))); do for n in real gauss; do
  f=$(printf "campaign/%s/%s/w%03d/summary.json" "$ev" "$n" "$w")
  [ -f "$f" ] || echo "$ev $w $n"
done; done; done | xargs -P "$NPAR" -L1 sh -c 'python pe_campaign.py --event $0 --window $1 --noise $2 > /dev/null 2>&1 || echo "FAIL $0 $1 $2" >> pe_failures.txt'
python collect_campaign.py --x fp
