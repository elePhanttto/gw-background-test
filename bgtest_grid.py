# fractional power のグリッドサーチ版
#claude製
# Macas & Lundgren (2023) PhysRevD.108.063016 の混合モデルを、
# NUTS ではなくパラメータ空間の全域走査で解く。
#
# 【NUTS ではなくグリッドにする理由】
#  - 推定するのは (F, sigma, nu) の3つだけ(lam を加えても4つ)。
#    数十次元ならグリッドは組合せ爆発するが、3-4次元なら全域走査が現実的。
#  - sigma と nu が強く縮退しており、勾配ベースの NUTS は細長い谷を延々と
#    さまよう。さらに識別性の問題で事後分布が多峰になると、NUTS は山を
#    飛び越えられず r_hat が悪化する(実データの 458-629Hz で 1.53-1.56)。
#    グリッドなら多峰でもそのまま複数の山として見える。
#  - グリッドの端に事後確率が溜まっていないかを確認でき、
#    「範囲が足りているか」を自分で検証できる。
#
# 【(sigma, nu) ではなく (mu_ng, nu) を掃く理由】
#  half Student-T の平均は mu_ng = sigma * c(nu) と sigma について線形なので、
#  sigma = mu_ng / c(nu) で一意に逆算できる。
#  mu_ng はタイルパワーと同じ単位なので、グリッド範囲をデータから直接決められる:
#    下限 = 識別性制約 (1.5 * mu_gauss)、上限 = データの99.9パーセンタイル程度。
#  一方 sigma を直に掃こうとすると、nu との縮退のせいで妥当な範囲が決めにくい。

import numpy as np
from scipy.special import gammaln
import pandas as pd


def c_nu(nu):
    """half Student-T の平均 / sigma。mu_ng = sigma * c(nu)。nu>1 で有限。
    nu=100 で 0.804、nu->無限大 で sqrt(2/pi)=0.798 とほぼ飽和するので、
    グリッドの上限は 100 程度で十分。"""
    return np.exp(np.log(2.0) + 0.5 * np.log(nu) + gammaln((nu + 1) / 2)
                  - 0.5 * np.log(np.pi) - gammaln(nu / 2) - np.log(nu - 1))


def half_t_logpdf(y, sigma, nu):
    """half Student-T (scale=sigma, df=nu) の対数確率密度 (y>=0)。"""
    return (np.log(2.0) + gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi) - np.log(sigma)
            - (nu + 1) / 2 * np.log1p(y ** 2 / (nu * sigma ** 2)))


def _binned(y, n_bins):
    """データを対数等間隔のビンにまとめる。
    素朴に「全グリッド点 x 全データ点」を計算すると N=8万で 10^10 回の演算に
    なるが、ビン化すると 1/100 程度に落ちる(実測: 3次元グリッドで約2秒)。
    裾が重いので、ビンは対数等間隔にし、代表値には幾何平均を使う。"""
    edges = np.geomspace(y.min() * 0.999, y.max() * 1.001, n_bins + 1)
    cnt, _ = np.histogram(y, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    keep = cnt > 0
    return cnt[keep], ctr[keep]


def grid_fp(y, lam_gauss=None, n_lam=25, n_mu=45, n_nu=25, n_F=60,
            n_bins=600, threshold_ratio=1.5, nu_max=100.0):
    """
    対数尤度をグリッド上で全部計算する。

    Parameters
    ----------
    y : ndarray
        正規化済みタイルパワー(norm='median' の q_gram 出力に ln2 を掛けたもの)
    lam_gauss : float or None
        None なら lam も4つ目のグリッド軸として一緒に推定する(推奨)。
        値を渡すとその値に固定して3次元グリッドになる。
        【なぜ lam も掃くか】中央値からの推定 lam = ln2/median は、
        非ガウス成分の混入率が高いと壊れる(混入30%で 0.688、真値1.0)。
        low quantile に変えても改善しきらない(Q10 で 0.749)。
        half Student-T はバルクにも重なるので、lam を単独でロバストに
        推定するのは原理的に難しく、同時推定した方が素直。
    threshold_ratio : float
        識別性制約。mu_ng >= threshold_ratio * mu_gauss をグリッドの下限にする。
        half Student-T は sigma,nu 次第で指数分布を模倣できてしまうので、
        「非ガウス成分はガウス成分よりパワーが大きい」を課して偽の解を除く。

    Returns
    -------
    dict : 対数尤度 LL と各グリッド軸
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    cnt, ctr = _binned(y, n_bins)

    if lam_gauss is None:
        lam0 = np.log(2.0) / np.median(y)
        lam_grid = np.geomspace(lam0 * 0.5, lam0 * 3.0, n_lam)
    else:
        lam_grid = np.array([float(lam_gauss)])

    nu_grid = np.geomspace(1.05, nu_max, n_nu)   # nu>1 が平均の有限条件
    F_grid = np.linspace(1e-4, 1 - 1e-4, n_F)
    mu_hi = max(y.mean() * 20.0, np.percentile(y, 99.9))

    LL = np.empty((len(lam_grid), n_mu, n_nu, n_F))
    MUs = np.empty((len(lam_grid), n_mu))

    for i, lam in enumerate(lam_grid):
        mu_g = 1.0 / lam
        mu_grid = np.geomspace(threshold_ratio * mu_g, mu_hi, n_mu)
        MUs[i] = mu_grid
        p1 = lam * np.exp(-lam * ctr)                       # ガウス成分(指数分布)
        MU, NU = np.meshgrid(mu_grid, nu_grid, indexing="ij")
        SIG = MU / c_nu(NU)                                 # sigma を逆算
        p2 = np.exp(half_t_logpdf(ctr[None, None, :], SIG[..., None], NU[..., None]))
        mix = ((1 - F_grid[None, None, :, None]) * p1[None, None, None, :]
               + F_grid[None, None, :, None] * p2[:, :, None, :])
        LL[i] = (cnt * np.log(np.maximum(mix, 1e-300))).sum(-1)

    return dict(LL=LL, lam_grid=lam_grid, MUs=MUs,
                nu_grid=nu_grid, F_grid=F_grid, n_tiles=len(y))


def summarize(r):
    """事後分布(一様事前分布)から fractional power をまとめる。

    edge_* はグリッドの端に溜まった事後確率。大きい場合は範囲不足なので、
    lam_grid / nu_max / mu_hi を広げて再計算すること。
    これは NUTS では得られない、グリッドならではの診断。
    """
    LL = r["LL"]
    w = np.exp(LL - LL.max())
    w /= w.sum()

    LAM = r["lam_grid"][:, None, None, None]
    MU = r["MUs"][:, :, None, None]
    F = r["F_grid"][None, None, None, :]
    mu_g = 1.0 / LAM
    fp = F * MU / ((1 - F) * mu_g + F * MU)
    fp = np.broadcast_to(fp, w.shape)          # nu 方向にも展開

    order = np.argsort(fp.ravel())
    fs, ws = fp.ravel()[order], w.ravel()[order]
    cdf = np.cumsum(ws)

    i, j, k, l = np.unravel_index(LL.argmax(), LL.shape)
    return dict(
        fp_mean=float((fp * w).sum()),
        fp_map=float(fp[i, j, 0, l]),
        fp_lo=float(np.interp(0.055, cdf, fs)),   # 89% 信用区間
        fp_hi=float(np.interp(0.945, cdf, fs)),
        lam_map=float(r["lam_grid"][i]),
        mu_ng_map=float(r["MUs"][i, j]),
        nu_map=float(r["nu_grid"][k]),
        F_map=float(r["F_grid"][l]),
        edge_lam=float(max(w.sum(axis=(1, 2, 3))[0], w.sum(axis=(1, 2, 3))[-1])),
        edge_mu=float(max(w.sum(axis=(0, 2, 3))[0], w.sum(axis=(0, 2, 3))[-1])),
        edge_nu=float(max(w.sum(axis=(0, 1, 3))[0], w.sum(axis=(0, 1, 3))[-1])),
        n_tiles=r["n_tiles"],
    )

def plot_loglik_contour(r,output,ax=None):
    """(mu_ng, nu) 平面の対数尤度を等高線で描く。
    多峰になっていないか(NUTS が r_hat 1.5 台で失敗した原因)を目で確認する用。
    lam と F は最尤点に固定して切る。"""
    from matplotlib import pyplot as plt
    LL = r["LL"]
    i, j, k, l = np.unravel_index(LL.argmax(), LL.shape)
    sl = LL[i, :, :, l]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(r["nu_grid"], r["MUs"][i], sl - sl.max(),
                     levels=np.linspace(-50, 0, 26))
    plt.colorbar(cs, ax=ax, label="log-likelihood (Difference from the maximum value)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nu"); ax.set_ylabel("mu_ng")
    ax.set_title(f"Cross-section at lam={r['lam_grid'][i]:.3f}, F={r['F_grid'][l]:.3f} ")
    fig.savefig(output)

def run_test_grid_freq(y,f_thinned,event,det,type,min_tiles=200):
    y = np.asarray(y, dtype=float)
    f_thinned = np.asarray(f_thinned, dtype=float)
    rows = []
    for f in np.unique(f_thinned):
        mask = (f_thinned == f)
        if mask.sum() < min_tiles:
            print(f"  f={f:.1f}Hz: タイル数 {mask.sum()} が min_tiles={min_tiles} 未満のためスキップ")
            continue
        # 辞書型のやつが返ってくる
        r = grid_fp(y[mask])
        s = summarize(r)
        s["freqs"] = f
        #s["type"] = type
        output = f"contour_{event}{det}_{f}_{type}.png"
        print(f"FP={s['fp_mean']:.4f} [{s['fp_lo']:.4f}-{s['fp_hi']:.4f}]  最尤点={s['fp_map']:.4f}")
        print(f"  lam={s['lam_map']:.3f}, mu_ng={s['mu_ng_map']:.2f}, nu={s['nu_map']:.1f}, F={s['F_map']:.3f}")
        print(f"  グリッド端の重み: lam={s['edge_lam']:.1e}, mu={s['edge_mu']:.1e}, nu={s['edge_nu']:.1e}")
        plot_loglik_contour(r,output)
        rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(f"{event}{det}_{type}_band_results_fracional_power_grid.csv",mode="a",index=False)
    #print(f"保存: {event}{det}_band_results_fracional_power.csv  ({len(df)}行)")