例の検定，割と簡単に作れそうなのでチョット仕様をまとめてみた

> 何をしているのかをコメントアウトで書いてもらう

1: 任意のイベントの前後の背景ノイズの正規化したq-transformed energyを計算する(GWPyのq-gram moduleを使う．4096秒間のうち，off-sourceな部分200箇所を等間隔に選ぶ👈️これは再現性のため)
2: 合成ガウスノイズも作っておく
3: 論文をベースにKSテスト(scipyのkstestモジュール)，ADテスト，カイ二乗検定を行う(ただし，計算資源の都合上，カイ二乗検定はサンプル数が10^6のため，p値の下限は10^(-6))
4: 横軸は時間(合体時刻からの差)・縦軸が各テストのp値(GW190412_adtest_H1.png/GW190412_chi2test_H1.png/GW190412_kstest_H1.png など)
5: kstestのp値の分布及び，q-transformed energyの分布もプロットする(GW190412_pvalue_hist_H1.png/GW190412_dist_check_H1など)
6: txtファイルは，q-transformed energyの分布について，裾(y>4)の部分が指数分布からどのくらい離れているか
7: ADテストでは，A^2統計量自体もプロットする(GW190412_a2_H1.pngなど)
8: 周波数ごとのq-transformedエネルギーの分布は，ホワイトニングがうまく行っているかどうかを見るためのものですが，今のところあまり気にしなくても良いと思います
9: q-transformed energyは時間方向に相関があるため(小さいp値がたくさん出る原因)，合成ガウスノイズをシミュレーションし，自己相関関数が0になる値を参考にタイルを間引いています!
10: `plot_bg...py`の自己相関関数についてはこちらの記事を参考にしてください: https://ja.wikipedia.org/wiki/%E8%87%AA%E5%B7%B1%E7%9B%B8%E9%96%A2
11: AIさんなどが読み込めるよう，結果をCSVファイルに出力するようにしています
12: 周波数ごとに同様のテストを行う`band`バージョンもあります．

As that test seems fairly straightforward to put together, I’ve summarised the specifications briefly.

> Please include a comment explaining what the code is doing

1: Calculate the normalised q-transformed energy of the background noise before and after an arbitrary event (using the q-gram module in GWPy. Select 200 off-source sections at regular intervals from the 4096-second period 👈️ This is for reproducibility)
2: Generate synthetic Gaussian noise as well
3: Based on the paper, perform the KS test (using the `kstest` module in SciPy), the AD test and the chi-squared test (however, due to computational resource constraints, the chi-squared test uses a sample size of 10⁶, so the lower bound for the p-value is 10⁻⁶)
4: The x-axis represents time (difference from the merger time) and the y-axis represents the p-values for each test (e.g. GW190412_adtest_H1.png, GW190412_chi2test_H1.png, GW190412_kstest_H1.png)
5: Plot the distribution of the p-values from the k-test and the distribution of the q-transformed energy (e.g. GW190412_pvalue_hist_H1.png, GW190412_dist_check_H1, etc.)
6: The txt file shows, with regard to the distribution of q-transformed energy, how far the tail (y > 4) deviates from an exponential distribution
7: For the AD test, plot the A² statistic itself as well (e.g. GW190412_a2_H1.png)
8: The distribution of q-transformed energy by frequency is intended to check whether the whitening process has been carried out correctly, but for the time being, I do not think it is necessary to pay too much attention to this
9: As q-transformed energy is correlated in the time domain (which causes a large number of small p-values to be produced), we simulate composite Gaussian noise and thin out the tiles based on the value at which the autocorrelation function becomes zero!
10: For details on the autocorrelation function in `plot_bg...py`, please refer to this article: https://ja.wikipedia.org/wiki/%E8%87%AA%E5%B7%B1%E7%9B%B8%E9%96%A2
11: I’ve set it up to output the results to a CSV file so that tools such as AI can read them.
12: You can test backgrounds deviding frequency into some BLOCKS with `background-test...band.py`

---

About blockwise version(Blockwise版について)

- PSDの変動を考慮して，ホワイトニングの期間を256秒・128秒・64秒・32秒に変えて同じ計算をしています
- Considering change of PSD, we calculate in the same way while changing whitening duration, 256 s,128 s,64 s,32 s.