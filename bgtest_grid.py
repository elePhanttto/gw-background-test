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

import os
import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t
import pandas as pd
import matplotlib.pyplot as plt

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
            n_bins=600, threshold_ratio=1.5, nu_max=100.0, nu_min=1.001,
            lam_range=(0.5, 3.0)):
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
    lam_range : (float, float)
        lam グリッドの範囲を lam0=ln2/median に対する比で指定する。
        既定 (0.5, 3.0) を n_lam=25 点で掃くと刻み幅が約7.75%になり、
        lam の統計精度(約0.5%)より桁違いに粗い。この粗さが「純粋なガウス
        ノイズでも fp が 0 にならない」主因だった。grid_fp_zoom() で2段階にする。
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
        lam_grid = np.geomspace(lam0 * lam_range[0], lam0 * lam_range[1], n_lam)
    else:
        lam_grid = np.array([float(lam_gauss)])

    # nu>1 が平均の有限条件。nu->1 に近づくほど mu_ng (平均) の意味が弱くなる。
    # nu_min=2.0 は旧版(下端に事後確率が張り付く行が出た)。実データでは 1.2 を推奨。
    nu_grid = np.geomspace(nu_min, nu_max, n_nu)
    F_grid = np.concatenate([[1e-4], np.geomspace(1e-3, 0.9999, n_F-1)]) #F を対数間隔
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
                nu_grid=nu_grid, F_grid=F_grid, n_tiles=len(y),
                cnt=cnt, ctr=ctr)


def grid_fp_zoom(y, margin=12.0, n_lam2=60,
                 coarse=dict(n_lam=25, n_mu=25, n_nu=12, n_F=30, n_bins=300), **kw):
    """lam を2段階で掃く階層グリッド(lam の刻みを粗さの原因から解放する)。

    1段目: 広い lam 範囲を、他の軸も粗くして素早く走査し、
           最大対数尤度から margin 以内に入る lam の範囲だけを残す。
    2段目: その狭い範囲を n_lam2 点で細かく走査し直す(他の軸は本来の解像度)。

    【なぜ必要か】lam の統計精度は約0.5%なのに、既定の lam グリッド刻みは約7.75%。
    格子の間に真の lam が落ちると、そのズレを非ガウス成分(F)が肩代わりして
    しまい、純粋なガウスノイズでも fp が約0.12になる。2段目の刻みは通常 0.5%
    以下になり、この人工的な底上げがほぼ消える。
    返り値は grid_fp と同じ dict。1段目で測ったグリッド端の重みを edge_lam に入れる
    (2段目の lam 範囲は意図的に狭いので、2段目の端の重みは診断にならない)。
    """
    r0 = grid_fp(y, **{**kw, **coarse})
    LL0 = r0["LL"]
    w0 = np.exp(LL0 - LL0.max()); w0 /= w0.sum()
    wl = w0.sum(axis=(1, 2, 3))
    keep = np.where((LL0 >= LL0.max() - margin).any(axis=(1, 2, 3)))[0]
    lg = r0["lam_grid"]
    lo = lg[max(keep.min() - 1, 0)]
    hi = lg[min(keep.max() + 1, len(lg) - 1)]
    yy = np.asarray(y, float)
    yy = yy[np.isfinite(yy) & (yy > 0)]
    lam0 = np.log(2.0) / np.median(yy)                # grid_fp 内部と同じ基準
    r = grid_fp(y, lam_range=(lo / lam0, hi / lam0), n_lam=n_lam2, **kw)
    r["edge_lam"] = float(max(wl[0], wl[-1]))
    r["lam_range_stage2"] = (float(lo), float(hi))
    return r


def summarize(r):
    """事後分布から fractional power をまとめる。

    事前分布は各軸の「対数一様」(grid の各点に等しい重み)。F は 1e-4〜1 を
    対数等間隔、mu_ng・nu・lam も対数等間隔なので、小さい F に事前質量が厚い。

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

    # 純指数分布(F=0)に対する対数尤度の改善量。0 に近いほど「混合しても得しない」。
    dlogL_exp = np.nan
    if "cnt" in r:
        cnt, ctr = r["cnt"], r["ctr"]
        lam_hat = cnt.sum() / (cnt * ctr).sum()
        ll_exp = (cnt * (np.log(lam_hat) - lam_hat * ctr)).sum()
        dlogL_exp = float(LL.max() - ll_exp)
    return dict(
        dlogL_exp=dlogL_exp,
        fp_mean=float((fp * w).sum()),
        fp_map=float(fp[i, j, 0, l]),
        fp_lo=float(np.interp(0.055, cdf, fs)),   # 89% 信用区間
        fp_hi=float(np.interp(0.945, cdf, fs)),
        fp_ul95=float(np.interp(0.95, cdf, fs)),     # 片側95%上限(fp>=0 なので下限が0付近のときはこちらを報告)
        lam_map=float(r["lam_grid"][i]),
        mu_ng_map=float(r["MUs"][i, j]),
        nu_map=float(r["nu_grid"][k]),
        F_map=float(r["F_grid"][l]),
        edge_lam=float(r.get("edge_lam", max(w.sum(axis=(1, 2, 3))[0], w.sum(axis=(1, 2, 3))[-1]))),
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
    #fig.savefig(output)

def grid_posterior_3d(r):
    """grid_fp の結果 r から (F, mu_ng, nu) の3次元事後分布を取り出す。

    LL の軸は (lam, mu, nu, F)。mu の格子は lam ごとに違う(下限が 1.5/lam)ので、
    最尤点の lam で切った断面(lam を固定した条件付き分布)を使う。
    grid_fp(y, lam_gauss=値) で lam を固定した場合は、厳密な周辺分布になる。
    """
    LL = r["LL"]
    i = np.unravel_index(LL.argmax(), LL.shape)[0]
    sl = LL[i]                          # (mu, nu, F)
    post = np.exp(sl - sl.max())
    post = post.transpose(2, 0, 1)      # -> (F, mu, nu)
    post /= post.sum()
    return post, r["F_grid"], r["MUs"][i], r["nu_grid"]


def corner_from_grid(post, F_grid, mu_grid, nu_grid, labels=None):
    """対角=1次元周辺分布、左下=2次元周辺分布。mu, nu は対数軸で描く。"""
    grids = [F_grid, mu_grid, nu_grid]
    logaxis = [False, True, True]       # mu, nu は geomspace なので対数軸
    labels = labels or [r"$F$", r"$\mu_{ng}$", r"$\nu$"]
    n = 3
    fig, axes = plt.subplots(n, n, figsize=(9, 9))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                marg = post.sum(axis=tuple(k for k in range(n) if k != i))
                ax.plot(grids[i], marg, color="C0")
                ax.fill_between(grids[i], marg, alpha=0.3, color="C0")
                ax.set_yticks([])
                if logaxis[i]:
                    ax.set_xscale("log")
            else:
                other = tuple(k for k in range(n) if k not in (i, j))
                m2 = post.sum(axis=other).T      # 残る軸は (j, i) の順 → 転置して (i, j)
                ax.pcolormesh(grids[j], grids[i], m2, cmap="viridis", shading="auto")
                if logaxis[j]:
                    ax.set_xscale("log")
                if logaxis[i]:
                    ax.set_yscale("log")
            if i == n - 1:
                ax.set_xlabel(labels[j])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])

    fig.tight_layout()
    return fig

def fp_samples(r, n_samp=200_000, seed=0):
    """4次元事後分布(lam, mu_ng, nu, F)から fp のサンプルを引く。

    【ノコギリ状ヒストグラムの原因と対策】
    fp は (lam, mu, F) の格子点でしか値を持たない(最大で数千個の離散値)。
    それを重み付きで100ビンに入れると、CI の幅が0.03程度のときは「値のある
    ビン」と「空のビン」が交互に並んでノコギリになる。そこで、格子点を事後確率に
    従って引き、各セルの内側で対数一様にばらす(格子の事前分布が対数一様なので整合的)。
    lam についても周辺化されるので、summarize の CI と同じ分布を表す
    (旧 fp_marginal は最尤 lam で切った条件付き分布だった)。"""
    rng = np.random.default_rng(seed)
    LL = r["LL"]
    w = np.exp(LL - LL.max()).ravel()
    w /= w.sum()
    idx = rng.choice(w.size, size=n_samp, p=w)
    i, j, k, l = np.unravel_index(idx, LL.shape)

    def jit(lg, step, ii):
        return np.exp(lg[ii] + rng.uniform(-0.5, 0.5, ii.size) * step[ii])

    lgF = np.log(r["F_grid"])
    F = np.clip(jit(lgF, np.gradient(lgF), l), 0.0, 1 - 1e-6)
    lgM = np.log(r["MUs"])                                   # (n_lam, n_mu)
    stepM = np.gradient(lgM, axis=1)
    mu = np.exp(lgM[i, j] + rng.uniform(-0.5, 0.5, i.size) * stepM[i, j])
    lam_g = r["lam_grid"]
    if len(lam_g) > 1:
        lgL = np.log(lam_g)
        lam = jit(lgL, np.gradient(lgL), i)
    else:
        lam = lam_g[i]
    return F * mu / ((1 - F) / lam + F * mu)


def fp_marginal(post, F_grid, mu_grid, nu_grid, mu_g, type, output1,freq, bins=100):
    F  = F_grid[:, None, None]
    mu = mu_grid[None, :, None]
    fp = F * mu / ((1 - F) * mu_g + F * mu)               # (F, mu, 1)
    fp = np.broadcast_to(fp, post.shape)                  # (F, mu, nu)
    hist, edges = np.histogram(fp.ravel(), bins=bins, range=(0, 1),
                               weights=post.ravel())
    centers = 0.5 * (edges[1:] + edges[:-1])
    cdf = np.cumsum(hist) / hist.sum()
    lo, hi = np.interp([0.055, 0.945], cdf, centers)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(centers, hist / hist.sum() / (edges[1] - edges[0]),
           width=edges[1] - edges[0], alpha=0.6, label=type)
    ax.axvspan(lo, hi, color="gray", alpha=0.15, label="89% CI")
    ax.set_xlabel("fractional power")
    ax.set_ylabel("posterior density")
    ax.set_title(f"fractional power posterior at freq={freq} Hz")
    ax.set_xlim((0.0,1.0))
    ax.legend()
    #fig.savefig(output1, dpi=150)
    plt.close(fig)                                        # ループで大量に作るのでメモリ対策
    return centers, hist, lo, hi

def fp_marginal_sampled(r, type, output1, freq, bins=100, n_samp=200_000, seed=0):
    """fp の周辺事後分布を描く(ノコギリ対策版)。fp_marginal の置き換え。"""
    fp = fp_samples(r, n_samp=n_samp, seed=seed)
    lo, hi, ul = np.percentile(fp, [5.5, 94.5, 95.0])
    x0, x1 = np.percentile(fp, [0.1, 99.9])
    pad = 0.1 * (x1 - x0) + 1e-3
    hist, edges = np.histogram(fp, bins=bins, range=(max(0.0, x0 - pad), min(1.0, x1 + pad)),
                               density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(centers, hist, width=edges[1] - edges[0], alpha=0.6, label=type)
    ax.axvspan(lo, hi, color="gray", alpha=0.15, label="89% CI")
    ax.axvline(ul, color="C3", ls="--", lw=1, label=f"95% upper limit = {ul:.3f}")
    ax.set_xlabel("fractional power")
    ax.set_ylabel("posterior density")
    ax.set_title(f"fractional power posterior at freq={freq} Hz")
    ax.set_xlim((0.0,1.0))
    ax.legend()
    #fig.savefig(output1, dpi=150)
    plt.close(fig)
    return centers, hist, lo, hi, ul


# ---------------------------------------------------------------------------
# 診断: 「混合モデルが当てはまる状況か」を fp の前に確認する
# ---------------------------------------------------------------------------
def tail_check(y, z_thr=3.0):
    """y の平均が純指数分布の期待値(=1)からどれだけズレているかで裾の重さを判定する。

    y は median=ln2 に規格化されているので、ガウス雑音なら E[y]=1, Var[y]=1。
    よって z=(mean-1)*sqrt(N) は N(0,1) に従う(タイルが独立なら)。
      z >  z_thr : "heavy"  (裾が重い → グリッチ的。fp が解釈できる状況)
      z < -z_thr : "light"  (裾が軽い → 線スペクトル等。混合モデルの想定外で fp は無意味)
    【なぜ必要か】Rice 分布(線スペクトル+ガウス雑音)の合成データで検証したところ、
    裾が軽いデータでも F->1, fp->1, 狭すぎる CI を返す(混合モデルが誤指定)。"""
    y = np.asarray(y, float); y = y[np.isfinite(y) & (y > 0)]
    z = (y.mean() - 1.0) * np.sqrt(len(y))
    tail = "heavy" if z > z_thr else ("light" if z < -z_thr else "ok")
    return dict(y_mean=float(y.mean()), y_mean_z=float(z), tail=tail)


def mixture_cdf(y, lam, mu_ng, nu, F):
    """混合分布 (1-F)Exp(lam) + F HalfT の累積分布関数。"""
    sig = mu_ng / c_nu(nu)
    return (1 - F) * (-np.expm1(-lam * y)) + F * (2.0 * student_t.cdf(y / sig, nu) - 1.0)


def ks_distance(y, cdf):
    """経験分布と cdf(y) の Kolmogorov-Smirnov 距離 D (スケールしていない生の値)。"""
    ys = np.sort(np.asarray(y, float))
    n = len(ys); c = cdf(ys)
    i = np.arange(1, n + 1)
    return float(max((i / n - c).max(), (c - (i - 1) / n).max()))


def diagnose_band(yb, s, event, det, type, f, tb=None, n_chunk=40, outdir="."):
    """帯域ごとの診断図(3枚組)を保存する。

    A: 生存関数 P(Y>y) の対数プロット。データ / 純指数 / 最尤混合モデル。
    B: Q-Q プロット(指数分布の分位点との比較)。右上で対角線より下 -> 裾が軽い、上 -> 重い。
    C: 時間チャンクごとの y の平均(tb を渡したとき)。1 からの系統的なズレ -> PSD の非定常。
       灰色帯は純ガウスなら 1±2/sqrt(n_chunk内のタイル数)。
    """
    yb = np.asarray(yb, float)
    ys = np.sort(yb)
    n = len(ys)
    sf = 1.0 - (np.arange(1, n + 1) - 0.5) / n
    ncol = 3 if tb is not None else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4.2))

    yy = np.geomspace(max(ys[0], 1e-3), ys[-1], 400)
    ax[0].loglog(ys, sf, ".", ms=2, color="k", label="data")
    ax[0].loglog(yy, np.exp(-yy), "C0--", lw=1.2, label="Exp(1) (pure Gaussian)")
    ax[0].loglog(yy, 1 - mixture_cdf(yy, s["lam_map"], s["mu_ng_map"], s["nu_map"], s["F_map"]),
                 "C3-", lw=1.2, label="MAP mixture")
    ax[0].set_ylim(max(1.0 / n / 3, 1e-8), 1.5)
    ax[0].set_xlabel("y (tile power, median=ln2)"); ax[0].set_ylabel("P(Y>y)")
    ax[0].legend(fontsize=8); ax[0].set_title("survival function")

    q_exp = -np.log(sf)
    ax[1].loglog(q_exp, ys, ".", ms=2, color="k")
    lim = [min(q_exp.min(), ys.min()), max(q_exp.max(), ys.max())]
    ax[1].loglog(lim, lim, "C0--", lw=1)
    ax[1].set_xlabel("Exp(1) quantile"); ax[1].set_ylabel("data quantile")
    ax[1].set_title(f"Q-Q  (tail={s.get('tail','?')}, mean={s.get('y_mean', np.nan):.3f})")

    if tb is not None:
        tb = np.asarray(tb, float)
        order = np.argsort(tb)
        chunks = np.array_split(order, n_chunk)
        tc = np.array([tb[c].mean() for c in chunks])
        mc = np.array([yb[c].mean() for c in chunks])
        nc = np.array([len(c) for c in chunks])
        ax[2].axhspan(1 - 2 / np.sqrt(nc.min()), 1 + 2 / np.sqrt(nc.min()), color="gray", alpha=0.2)
        ax[2].plot(tc, mc, "o-", ms=3)
        ax[2].axhline(1.0, color="C0", ls="--", lw=1)
        ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("chunk mean of y")
        ax[2].set_title("band power vs time")

    fig.suptitle(f"{event} {det} {f} Hz  fp={s['fp_mean']:.3f}  status={s.get('status','')}", fontsize=10)
    fig.tight_layout()
    #fig.savefig(os.path.join(outdir, f"diag_{event}{det}_{f}_{type}.png"), dpi=130)
    plt.close(fig)


def run_test_grid_freq(y, f_thinned, event, det, type, min_tiles=200, zoom=True,
                       nu_min=1.001, t_thinned=None, diag=True, outdir="."):
    """
    帯域ごとに fractional power を推定し、CSV に追記する。

    新しい引数
    ----------
    nu_min : ν グリッドの下端。既定 1.2(旧版は 2.0)。旧結果と比べるなら 2.0 を渡す。
    t_thinned : y と同じ長さの時刻配列。渡すと診断図 C(時間チャンク平均)が付く。
    diag : True で帯域ごとに診断図 diag_*.png を保存する。
    CSV の追加列:
      y_mean, y_mean_z, tail : 裾の重さの判定(tail_check)
      ks_exp, ks_fit         : KS 距離(純指数 / 最尤混合)。ks_fit が ks_exp より十分小さいときだけ混合の意味がある
      dlogL_exp              : 純指数に対する最大対数尤度の改善量
      status                 : "" / "light-tail(model mismatch)" / "unidentified(bimodal)"
    """
    y = np.asarray(y, dtype=float)
    f_thinned = np.asarray(f_thinned, dtype=float)
    t_arr = None if t_thinned is None else np.asarray(t_thinned, dtype=float)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for f in np.unique(f_thinned):
        mask = (f_thinned == f)
        if mask.sum() < min_tiles:
            print(f"  f={f:.1f}Hz: タイル数 {mask.sum()} が min_tiles={min_tiles} 未満のためスキップ")
            continue
        yb = y[mask]
        r = (grid_fp_zoom(yb, nu_min=nu_min) if zoom else grid_fp(yb, nu_min=nu_min))
        s = summarize(r)
        s["freqs"] = f
        s.update(tail_check(yb))
        ok = np.isfinite(yb) & (yb > 0)
        s["ks_exp"] = ks_distance(yb[ok], lambda v: -np.expm1(-v))
        s["ks_fit"] = ks_distance(yb[ok], lambda v: mixture_cdf(
            v, s["lam_map"], s["mu_ng_map"], s["nu_map"], s["F_map"]))

        post, Fg, Mg, Ng = grid_posterior_3d(r)
        fig = corner_from_grid(post, Fg, Mg, Ng)
        fig.suptitle(f"{event} {det}  f={f}Hz")
        #fig.savefig(os.path.join(outdir, f"corner_{event}{det}_{f}_{type}.png"), dpi=150)
        plt.close(fig)
        output1 = os.path.join(outdir, f"fp_merginal_{event}{det}_{f}_{type}.png")
        centers, hist, lo, hi, ul = fp_marginal_sampled(r, type, output1, f)
        s["fp_ul95"] = float(ul)

        status = []
        if s["tail"] == "light":
            status.append("light-tail(model mismatch)")
        if s["fp_hi"] - s["fp_lo"] > 0.5:
            status.append("unidentified(bimodal)")
        s["status"] = ";".join(status)

        print(f"{f:g}Hz FP={s['fp_mean']:.4f} [{s['fp_lo']:.4f}-{s['fp_hi']:.4f}]  95%上限={s['fp_ul95']:.4f}"
              f"  (最尤点={s['fp_map']:.4f})  tail={s['tail']} (mean={s['y_mean']:.3f}, z={s['y_mean_z']:+.1f})"
              f"  KS exp/fit={s['ks_exp']:.4f}/{s['ks_fit']:.4f}")
        print(f"  lam={s['lam_map']:.3f}, mu_ng={s['mu_ng_map']:.2f}, nu={s['nu_map']:.1f}, F={s['F_map']:.3f}"
              f"  グリッド端の重み: lam={s['edge_lam']:.1e}, mu={s['edge_mu']:.1e}, nu={s['edge_nu']:.1e}")
        if s["status"]:
            print(f"  ⚠ {s['status']}: この行の fp は数値として使わないでください")
        if diag:
            diagnose_band(yb, s, event, det, type, f,
                          tb=None if t_arr is None else t_arr[mask], outdir=outdir)
        rows.append(s)

    df = pd.DataFrame(rows)
    path = os.path.join(outdir, f"{event}{det}_{type}_band_results_fracional_power_grid.csv")
    # 帯域グループごとに呼ばれて追記されるので、ヘッダはファイルが無いときだけ書く
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    return df

def thin_per_row(e_all, f_all, t_all, idx, rate):
    """周波数行ごとに時刻順で間引く (行の取りこぼしが起きない)"""
    keep = []
    for f in np.unique(f_all[idx]):
        r = idx[f_all[idx] == f]
        r = r[np.argsort(t_all[r], kind="stable")]
        keep.append(r[::rate])
    return np.concatenate(keep)

# ---------------------------------------------------------------------------
# 局所PSDホワイトニング版 (PE と同じ「窓の直前 128 s の PSD」でホワイトニングする)
# ---------------------------------------------------------------------------
# 【なぜ必要か】従来版は 4096 s 全体の PSD で一括ホワイトニングしている。
#  PE (bilby) は窓の直前 128 s から PSD を推定するので、ゆっくりした非定常性は
#  PE 側では PSD に吸収される。PE が実際に見ている非ガウス性を測るには、
#  背景テストのホワイトニング条件も PE に揃える必要がある。
PSD_SPAN_LOCAL = 128.0   # PSD 推定に使う長さ [s] (pe_campaign.py の PSD_SPAN と同じ)
PSD_GAP_LOCAL = 2.0      # 窓と PSD 区間の隙間 [s]
PAD_LOCAL = 4.0          # ホワイトニングの端の影響を避ける余白 [s] (>= fduration/2)


def local_white_segment(data, t, seg_duration, psd_span=PSD_SPAN_LOCAL, psd_gap=PSD_GAP_LOCAL,
                        pad=PAD_LOCAL, fftlength=4, overlap=2, fduration=4):
    """窓 [t, t+seg_duration] を、直前 psd_span 秒の median PSD でホワイトニングして返す。
    PSD 区間か余白がデータ外なら None。"""
    t0, t1 = float(data.span[0]), float(data.span[1])
    if t - psd_gap - psd_span < t0 or t + seg_duration + pad > t1 or t - pad < t0:
        return None
    psd = data.crop(t - psd_gap - psd_span, t - psd_gap).psd(
        fftlength=8, overlap=4, method="median", window=("tukey", 0.1))   # PE と同じ設定に
    ext = data.crop(t - pad, t + seg_duration + pad)
    w = ext.whiten(asd=psd ** 0.5, fduration=fduration)
    return w.crop(t, t + seg_duration)


def collect_local_tiles(data, random_time, seg_duration, Q, BANDS):
    """全窓 x 全帯域の Q タイルを局所PSDホワイトニングで集める。
    窓ごとにホワイトニングは1回だけ行い、帯域ごとに q_gram を取る。
    返り値: e[i], f[i], t[i] (帯域 i ごとの配列), used (使えた窓の開始時刻)"""
    nb = len(BANDS)
    e_l = [[] for _ in range(nb)]; f_l = [[] for _ in range(nb)]; t_l = [[] for _ in range(nb)]
    used = []
    for t in random_time:
        w = local_white_segment(data, t, seg_duration)
        if w is None:
            continue
        used.append(t)
        for i, fmin, fmax in BANDS:
            qg = w.q_gram(qrange=[Q, Q], frange=[fmin, fmax], snrthresh=0)
            e_l[i].append(np.asarray(qg["energy"]))
            f_l[i].append(np.asarray(qg["frequency"]))
            t_l[i].append(np.asarray(qg["time"]))
    e = [np.concatenate(x) for x in e_l]
    f = [np.concatenate(x) for x in f_l]
    tt = [np.concatenate(x) for x in t_l]
    print(f"局所PSD版: 使えた窓 {len(used)}/{len(random_time)}")
    return e, f, tt, np.asarray(used)


def run_test_localpsd(data, random_time, seg_duration, Q, BANDS, rate_of_mabiki,
                      event, det, type="real_localpsd", y_cut=100.0, seg_mean_cut=None,
                      geocent_time=None):
    """局所PSD版の背景テスト。
    1) 全帯域のタイルを集める
    2) y > y_cut のタイルを含む窓を「過渡ノイズ窓」として除く (全帯域の和集合)
    3) seg_mean_cut を与えると、窓平均 y がそれを超える窓も「エピソード窓」として除く
    4) 周波数行ごとに時刻順で間引いて run_test_grid_freq に渡す
    CSV の type は  {type}_all (除外なし) と {type}_clean (除外後) の2種類。"""
    e, f, tt, rt = collect_local_tiles(data, random_time, seg_duration, Q, BANDS)

    bad = set(); epi = set()
    for i, fmin, fmax in BANDS:
        y_all = e[i] * np.log(2.0)
        seg_all = np.searchsorted(rt, tt[i], side="right") - 1
        bad |= set(np.unique(seg_all[y_all > y_cut]).tolist())
        if seg_mean_cut is not None:
            for s in range(rt.size):
                m = seg_all == s
                if m.any() and y_all[m].mean() > seg_mean_cut:
                    epi.add(s)
    bad = sorted(bad); epi = sorted(epi - set(bad))
    t_ref = 0.0 if geocent_time is None else geocent_time
    print(f"[{type}] 過渡ノイズ窓: {len(bad)}/{rt.size}  "
          + ", ".join(f"{rt[s]-t_ref:+.1f}s" for s in bad))
    if seg_mean_cut is not None:
        print(f"[{type}] エピソード窓 (窓平均y>{seg_mean_cut}): {len(epi)}/{rt.size}")

    for i, fmin, fmax in BANDS:
        seg_all = np.searchsorted(rt, tt[i], side="right") - 1
        idx_all = thin_per_row(e[i], f[i], tt[i], np.arange(e[i].size), rate_of_mabiki[i])
        run_test_grid_freq(e[i][idx_all] * np.log(2.0), f[i][idx_all], event, det,
                           type=f"{type}_all")
        drop = list(bad) + list(epi)
        idx = np.where(~np.isin(seg_all, drop))[0]
        idx = thin_per_row(e[i], f[i], tt[i], idx, rate_of_mabiki[i])
        run_test_grid_freq(e[i][idx] * np.log(2.0), f[i][idx], event, det,
                           type=f"{type}_clean")
    return dict(rt=rt, bad=bad, episode=epi)