例の検定，割と簡単に作れそうなのでチョット仕様をまとめてみた

> 何をしているのかをコメントアウトで書いてもらう

1: GW190412・GW190828...のH1/L1(少なくとも大きなグリッチがあるとは言及されてない)について，イベントの前後の背景ノイズの正規化したq-transformed energyを計算する(GWPyのq-gram moduleを使う．4096秒間のうち，off-sourceな部分200箇所を等間隔に選ぶ👈️これは再現性のため)
2: 論文をベースにKSテスト(scipyのkstestモジュール)，ADテスト，カイ二乗検定を行う(ただし，計算資源の都合上，カイ二乗検定はサンプル数が10^6のため，p値の下限は10^(-6))
3: 横軸は時間(合体時刻からの差)・縦軸が各テストのp値(GW190412_adtest_H1.png/GW190412_chi2test_H1.png/GW190412_kstest_H1.png など)
4: kstestのp値の分布及び，q-transformed energyの分布もプロットする(GW190412_pvalue_hist_H1.png/GW190412_dist_check_H1など)
5: txtファイルは，q-transformed energyの分布について，裾(y>4)の部分が指数分布からどのくらい離れているか
6: ADテストでは，A^2統計量自体もプロットする(GW190412_a2_H1.pngなど)
7: 周波数ごとのq-transformedエネルギーの分布は，ホワイトニングがうまく行っているかどうかを見るためのものですが，今のところあまり気にしなくても良いと思います
8: q-transformed energyは時間方向に相関があるため(小さいp値がたくさん出る原因)，合成ガウスノイズをシミュレーションし，自己相関関数が0になる値を参考にタイルを間引いています!
9: `plot_bg...py`の自己相関関数についてはこちらの記事を参考にしてください: https://ja.wikipedia.org/wiki/%E8%87%AA%E5%B7%B1%E7%9B%B8%E9%96%A2
10: AIさんなどが読み込めるよう，結果をCSVファイルに出力するようにしています

As that test seems fairly straightforward to put together, I’ve summarised the specifications briefly.

> Please include a comment explaining what the code is doing

1: For the H1/L1 events in GW190412, GW190828, etc. (for which at least no major glitches have been reported), calculate the normalised q-transformed energy of the background noise before and after the event (using the q-gram module in GWPy. Select 200 off-source sections at regular intervals from the 4096-second period 👈️ This is for reproducibility)
2: Based on the paper, perform a KS test (using the `kstest` module in `scipy`), an AD test and a chi-squared test (however, due to computational resource constraints, the chi-squared test has a sample size of 10⁶, so the lower bound of the p-value is 10⁻⁶)
3: The x-axis represents time (difference from the merger time) and the y-axis represents the p-values for each test (e.g. GW190412_adtest_H1.png, GW190412_chi2test_H1.png, GW190412_kstest_H1.png)
4: Plot the distribution of the p-values from the k-test and the distribution of the q-transformed energy (e.g. GW190412_pvalue_hist_H1.png, GW190412_dist_check_H1, etc.)
5: The txt file shows, for the distribution of the q-transformed energy, how far the tail (y > 4) deviates from an exponential distribution
6: For the AD test, plot the A² statistic itself as well (e.g. GW190412_a2_H1.png)
7: The distribution of q-transformed energy by frequency is intended to check whether the whitening process has been carried out correctly, but for the time being, I do not think you need to worry about it too much
8: Since the q-conversion energy exhibits temporal correlation (which is why so many small p-values occur), we simulate synthetic Gaussian noise and thin out the tiles based on the value at which the autocorrelation function becomes zero!
9: About "Autocorrelation" in `plot_bg...py`,you can see : https://en.wikipedia.org/wiki/Autocorrelation
10: A `.csv` file is exported to enable AI to read figures