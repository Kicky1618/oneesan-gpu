# GPU 数え上げおねえさん — Manim 解説動画

今回の GPU 実装を、約 60 秒で説明する Manim Community Edition の動画です。

## 内容

1. 数え上げおねえさん問題 — 格子上の自己回避路の総数
2. 愚直 DFS と GPU divergence
3. frontier DP — 過去全体ではなく境界状態だけを保持
4. 2-bit packed MateID
5. rank / unrank による dense `uint32[]` 化
6. transition-closed group を 10–14 遷移まとめる specialized CUDA kernel
7. RTX 5090 の実測値
   - n=18: 0.766 s
   - n=20: 10.749 s
   - n=21: 34.984 s
   - n=22: 120.838 s
   - n=23: 302.634 s
   - n=23 peak: 29.481 GiB
   - residue: 2762394459
8. B300 × 8 を想定した multi-GPU state sharding

## Render

低品質プレビュー:

```bash
uv run manim -pql src/manimations/oneesan_gpu.py OneesanGPUExplainer
```

1080p:

```bash
uv run manim -pqh src/manimations/oneesan_gpu.py OneesanGPUExplainer
```

キャッシュを使わず全シーンを検証:

```bash
uv run manim -ql --disable_caching src/manimations/oneesan_gpu.py OneesanGPUExplainer
```

出力先は通常 `media/videos/oneesan_gpu/` です。

## Source

- `src/manimations/oneesan_gpu.py`

Manim は `>=0.19.0,<0.20` に固定しています。日本語フォントには `Noto Sans CJK JP` / `Noto Sans Mono CJK JP` を使用します。

## 3b1b-inspired version

UI-card styleを避け、黒背景・細い格子・色付き数理オブジェクト・連続変形を中心にした版:

```bash
uv run manim -pqh src/manimations/oneesan_gpu_3b1b.py OneesanGPU3B1B
```

- `src/manimations/oneesan_gpu_3b1b.py`
- 約75秒
- LaTeX環境に依存せず、数式相当の表示は Noto Sans Mono + Unicode で描画
