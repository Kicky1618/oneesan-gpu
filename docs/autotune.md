# GPU構成を検出してGROUPBATCHを自動調整する

```bash
python3 scripts/run/autotune.py 27
```

起動時に`autotune implementation: direct GROUPBATCH nvcc build`と表示されることを確認する。
`b300-hbm32-batch.sh`がログに現れる場合は古いautotuneスクリプトなので、リポジトリの更新後に再実行する。

CUDA Toolkitの`nvcc`と`nvidia-smi`が必要。対象は今回最適化したGROUPBATCH経路の
n=20〜27。GPUを検出し、必要なバイナリをビルドして、測定・設定選択の後に
既存のチェックポイント付きCRTソルバーを実行する。GPUのレンタルは行わない。

検出するのはCUDAから見えるGPUのUUID、Compute Capability、SM数、総VRAM・空きVRAM、
VMM粒度、ドライバ、GPU間の双方向P2P可否。検出プログラムはカーネルを起動しないが、
空きメモリ取得のためCUDAコンテキストを作る。MIGは対象外。
`CUDA_VISIBLE_DEVICES`で使用可能なGPUを限定できる。

B300のCompute Capability 10.3では、nvccが`sm_103`を持っていればそれを使う。
CUDA 12.8のように持っていない場合は、`sm_100f`が対応していれば10.xファミリー互換
ターゲットとして使う。どちらもなければ、CUDA Toolkit 12.9以上が必要だと明示して停止する。

```bash
# 検出のみ。ソルバーのビルド・ベンチマーク・本実行はしない
python3 scripts/run/autotune.py --detect-only

# 指定したGPUだけを使い、設定選択まで行う
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run/autotune.py 21 --tune-only

# ベンチマーク時間を拡大し、本番と同じn=21で測定する
python3 scripts/run/autotune.py 21 --bench-n 21 --budget-seconds 900

# 一致する設定が保存済みなら再利用し、CRT計算を再開する
python3 scripts/run/autotune.py 27 --reuse --work-dir work/exact-auto-n27
```

探索する設定は次のとおり。

| 項目 | 候補 |
|---|---|
| GPU構成 | 同じCompute Capabilityで完全P2P接続がある1〜8台 |
| scratch容量 | 512 / 2048 / 10240 MiB（`--scratch-mib`で変更可能） |
| 交互配置の列数 | 1 / 8 / 16 / 32 |
| スレッド数 | 128 / 256 / 512 |
| CUDA Graph | 0 / 1 / 2 |
| 動的分配 | 静的 / affinity・sticky・reclaim付き動的分配 |

同じ台数・Compute Capabilityでは、最小空きVRAMが最大のP2Pグループを候補とする。
全GPU部分集合や全設定の直積を総当たりする方式ではない。
最初にGPU構成とscratch容量を比較し、その勝者から列数、スレッド数、Graph、分配方式、
列数を順に変えて改善する候補を探す。他のバックエンドや半幅分割の変更は探索しない。

**既定のベンチマークはn=20で、本番サイズが大きい場合は代理測定となる。**
大規模問題での真の最適設定を保証するものではなく、探索して完走した候補の中で
最速の設定を選ぶ。特にNVLink通信の割合やscratchの効果はサイズによって変わる。
本番サイズで比較する場合は`--bench-n`を本番のnと等しくする。

各候補は1回ウォームアップ後、既定3回の`wall_s`の中央値で比較する。
`wall_s`は共通の初期化等を除いたDP時間で、プロセス全体の時間も別途記録する。
n=20/21では既知の剰余値を確認し、それ以外では最初の成功実行との一致を確認する。
剰余不一致は探索全体を中止する。OOM、実行失敗、タイムアウトの候補は採用しない。
タイムアウトやCtrl-Cでは子プロセス群も終了させる。

既定の探索予算は600秒、1標本の上限は90秒。初期ビルド・初期メモリ計画の時間は
この予算に含めない。候補の途中で予算が尽きた場合、その候補を採用せず、既に全反復が
完了した候補から選ぶ。完走した候補がなければ本実行を開始しない。

各候補は検証済みGROUPBATCHソースを直接nvccでビルドし、本番・ベンチマーク両サイズの`GRIDFP_PLAN_ONLY`を通し、scratchに収まること、
row6初期化とDP段階の両ピークが全GPUの空きVRAM以下であることを確認する。
既定の余裕は1 GiB/GPUに丸め誤差用16 MiBを加えた値。
`--reserve-mib 8192`などで増やせる。本実行直前にも空きVRAMを再取得する。
CUDA Graphやドライバの計画外メモリ消費を完全に予測するものではない。

設定・測定記録は既定で`work/autotune/n27.json`、各標本とメモリ計画の生ログは
同じ場所の実行別ディレクトリへ保存する。`--output`で変更可能。
`--reuse`はGPU UUID・構成・ドライバ・CUDA Toolkit・ソース検証済みバイナリ・
本スクリプト・問題サイズ・主要な探索条件が一致するときだけ設定を再利用する。
空きVRAMは毎回再検査する。バイナリとビルド来歴は`build/autotune/`に保存する。通常の実験用ビルドスクリプトにあるrow-limit等のソース変換は経由しない。

測定時と本実行では、親環境の`GRIDFP_*`や実験用のROW6/7/8設定を引き継がず、
選択した設定を明示する。`GRIDFP_PLAN_ONLY=1`の残留などで本計算が省略されることを防ぐ。
並行するGPU負荷や温度・クロック変化は測定順位に影響するため、再調整時は他の計算を
止めた環境で実行する。同じ出力先の同時調整・本実行はロックで拒否する。
共有するバイナリのビルドもロックで直列化する。

CPUのみの検証:

```bash
python3 -m unittest discover -s tests -p test_autotune.py -v
```

GPU構成の選別、メモリ判定、誤答・タイムアウトの扱い、キャッシュの条件、
本実行への設定引き渡しを模擬ソルバーで検証する。実GPUでの速度や最適性の検証とは区別する。
