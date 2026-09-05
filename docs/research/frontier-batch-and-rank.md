# 同型フロンティアの一括転送と線形圧縮の限界

2026-09-05。対象は `oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu`。
GPU の変更は `factor_transfer_batch.cuh`、独立照合は
`tests/factor_transfer_batch_test.cu` にある。

## 同じ演算子を複数の値ベクトルへ適用する

一つのウィンドウ内では、非活動領域の占有マスクが異なるグループは互いに遷移しない。
占有数が等しいマスク間には、占有記号の順序を保って N の位置だけを移す全単射がある。
局所 rank と遷移の多重度を保つことは
[前回の証明](../results/frontier-compiled-2026-09-05.md) に記した。

通常・blocked の値を合わせたグループ g の列を x_g、共通の整数転送行列を T とする。
X = [x_0 x_1 ... x_(b-1)] と並べると、更新は **Y = TX** である。
これは b 個の独立した更新と成分ごとに等しい。
状態をグループ優先に並べた行列は I_b ⊗ T、状態優先なら T ⊗ I_b になり、
両者は添字の置換で結ばれる。各列の値は異なってよく、同一視も平均化もしない。

実装では L 列ずつまとめ、添字を

```
address(group, state) = floor(group/L) * state_count * L
                      + state * L + (group mod L)
```

とする。L は 1, 2, 4, 8, 16, 32 のいずれかで、一つのバッチ内では固定。
同じ warp 内で同じ出力状態を処理するスレッドが増え、CSR の行・列番号を同じアドレスから
読み、入力値を隣接アドレスから読む。出力行ごとの逆像数の差による分岐のばらつきも減る。
これはハードウェアカウンターでの帯域飽和の証明ではない。

L の既定上限は 16、バッチの既定上限は 32 グループ。
利用可能な scratch から L を小さくし、L の倍数のグループ数だけ収容する。
端数バッチは L の倍数まで領域を確保するが、実在しない列はカーネル冒頭で除外する。
通常・blocked それぞれの入力、出力、64-bit グローバル番号を保存するため、
パディング後の b に対して必要量は
`16*b*(main_count + blocked_count)` bytes と各領域の整列分である。
余分なバッチ領域は既存の scratch 上限内で確保する。

## 最後の転送と書き戻しの融合

各グループのローカル番号 i からグローバル番号への写像を G_g(i) とする。
gather 時に G_g(i) をキャッシュし、最後の転送は局所出力配列を経ず、直接
`global[G_g(i)] = (T_last x_g)[i]` と書く。

G_g は各セクターのグループ内で単射であり、異なる占有マスクの像は交わらない。
最後の転送が読む配列は scratch にあり、書き込む authoritative 配列と別領域である。
従って同時実行による入力の上書きや書き込み衝突は起こらない。
次のバッチは同じストリームで先行バッチの完了後に実行する。
ウィンドウ間では全 GPU worker を同期してから、別のグループ分割へ移る。

CUDA graph のキーには状態数・窓・配置・実グループ数を含める。
マスク列の転送、factor 設定の転送、graph 実行は同じストリームに並べる。
マスク列を持つホスト schedule は全剰余の処理が終わるまで生存する。
arena の拡張、転送表の解放時には、旧ポインターを参照する graph を破棄する。

## 任意の接続状態を線形圧縮する場合の障壁

これは既存の meander determinant の定理からの帰結であり、新しい一般計数アルゴリズム
という主張ではない。有限の正方格子で実際に到達する部分空間の次元下界でもない。

2k 点上の非交差完全マッチングの集合を A_k とする。大きさは Catalan 数 C_k。
a と b を上下から貼り合わせた閉路数を c(a,b) とし、

```
G_k(q)[a,b] = q^c(a,b)
M_k[a,b]   = 1 if c(a,b) == 1 else 0
```

と定義する。M_k は「一つの閉路になる接続だけを数える」行列であり、
G_k(q) = q M_k + O(q²) である。

Di Francesco の [Meander Determinants, Theorem 1](https://arxiv.org/abs/hep-th/9612026)
によれば、U_0=1, U_1=q, U_(j+1)=q U_j-U_(j-1) としたとき、

```
det G_k(q) = product(j=1..k) U_j(q)^a_j
a_j = binom(2k,k-j) - 2 binom(2k,k-j-1) + binom(2k,k-j-2).
```

U_(2r)(0)=(-1)^r、U_(2r+1)(q)=(-1)^r(r+1)q+O(q³) である。
また奇数 j にわたる a_j の和は C_k になる。b(t)=binom(2k,t) と書けば、
その和は `2 sum(t=0..k-1) (-1)^(k-1-t) b(t) - b(k-1)`。
交代二項和の恒等式から `2 binom(2k-1,k-1)-binom(2k,k-1)=C_k` を得る。

従って det G_k の q に関する最低次数はちょうど C_k で、その係数は

```
det M_k = product(j even) ((-1)^(j/2))^a_j
        * product(j odd)  ((-1)^((j-1)/2)*(j+1)/2)^a_j != 0.
```

a_j は負になることがあるが、各因子は非零なので有理関数としての最低次項の計算は有効。
この結果は整数行列 M_k の非零な整数行列式を与える。
素数 p > ceil(k/2) なら全因子が法 p でも可逆なので、M_k はその体上でも正則である。

任意のマッチング係数と任意の接続先について答えを保存する線形符号化を d 次元に
圧縮できるなら M_k が d 次元を経由して因数分解され、rank M_k ≤ d となる。
よって **d ≥ C_k**。q=0 で G_k 自体が零になることを理由に、単一閉路計数の
接続情報を消すことはできない。必要なのは q の一次係数である。

この障壁は、境界条件・残りの行数・到達可能部分空間に特化した圧縮、非線形表現、
別の分割法を排除しない。既存の固定行数オートマトンを改良する余地は残る。
`scripts/tools/meander_rank_obstruction.py` はマッチングを直接生成し、グラフの連結性から
M_k を作り、有限体 Gaussian elimination の結果をこの式と照合する。

## 既存の枝刈り手法との適用範囲の違い

Jensen の [self-avoiding walk 転送法](https://arxiv.org/abs/1309.6709)、§2.2 は、
長さ上限に収まらない部分状態を除くため、将来の接続を保持して必要な追加長を計算する。
本ソルバーは正方格子内の全長の単純路を数えるため、その長さ上限に依存する枝刈りを
そのまま移植できない。将来の幾何制約を使うには、別途この計数問題に対応する証明が必要。

## 再現方法

### n=27 への適用範囲

小規模の倍率は n=27 に外挿できない。
`scripts/tools/frontier_batch_capacity.py` は N を許す位置と占有必須位置の高さ DP から
各占有数のグループサイズを整数で計算し、各サイズを二項係数で重み付けする。
全グループの合計が自由フロンティアの状態数と一致することも検査する。

n=27、LOW=14、HIGH=13 の通常状態は 385,719,506,620、blocked は
135,015,505,407 個である。scratch 16 GiB、表 512 MiB の既定上限では、
現行の 32-bit scan の入場条件と CSR のオフセットだけを考慮しても、バッチ候補の
状態数の割合は上側ウィンドウで **高々18.53%**、下側で **高々4.74%**。
この上限には CSR の列番号や構築用一時領域を含めていないため、実際の適用率はさらに低い。
これはカーネル時間の割合ではなく、各グループの通常・blocked 状態数を合計した割合である。

この制限を解くには、全状態サイズの CSR を持たない局所因子の転送や、十分な予算のある
GPU 向けの別の構築方式が必要になる。今回の n=20/21 の改善だけで n=27 の主要な
計算時間が解決したとはいえない。

### コマンド

```
python scripts/bench/bench_factor_division.py --optimization transfer-batch --n 20 --arch sm_86 --repeats 6
python scripts/bench/bench_factor_division.py --optimization transfer-batch --n 21 --modulus 4294966997 --arch sm_86 --repeats 4
python scripts/tools/meander_rank_obstruction.py --max-pairs 7
python scripts/tools/frontier_batch_capacity.py --n 27 --scratch-mib 16384 --map-mib 512
scripts/test/gridfp-reverse.sh --gpu
```

`GRIDFP_TRANSFER_BATCH=1` は従来の一グループ経路。
転送表が入らない場合も従来経路に戻る。
`GRIDFP_PROFILE_BATCH=1` は CUDA event で gather、中間転送、最終転送と書き戻しを計測する。
このモードはバッチで graph を使わず、各バッチを同期するため、
通常実行の wall time と比較する用途には使わない。
