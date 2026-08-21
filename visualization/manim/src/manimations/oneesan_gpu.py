from __future__ import annotations

from dataclasses import dataclass

from manim import (
    AnimationGroup,
    Arrow,
    BLACK,
    BLUE,
    BLUE_D,
    BLUE_E,
    Brace,
    Circle,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    Flash,
    GREEN,
    GREEN_C,
    GREEN_E,
    GREY_B,
    GREY_D,
    GrowArrow,
    GrowFromCenter,
    GrowFromPoint,
    Group,
    LaggedStart,
    LEFT,
    Line,
    ORANGE,
    ORIGIN,
    PI,
    PURPLE,
    Rectangle,
    RED,
    RIGHT,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    TransformFromCopy,
    UP,
    VGroup,
    WHITE,
    Write,
    YELLOW,
    config,
)

config.background_color = "#0b1020"

JP_FONT = "Noto Sans CJK JP"
MONO_FONT = "Noto Sans Mono CJK JP"

BG = "#0b1020"
FG = "#f5f7ff"
MUTED = "#aab4cc"
ACCENT = "#5aa9ff"
CYAN = "#55d6e8"
LIME = "#81e6a8"
WARN = "#ffbd66"
HOT = "#ff6b7a"
PANEL = "#161d31"
PANEL_2 = "#202944"


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    spacing: float = 0.62


def txt(s: str, size: int = 32, color=FG, *, bold: bool = False, mono: bool = False) -> Text:
    return Text(
        s,
        font=MONO_FONT if mono else JP_FONT,
        font_size=size,
        color=color,
        weight="BOLD" if bold else "NORMAL",
    )


def panel(width: float, height: float, *, color=PANEL, radius: float = 0.18) -> RoundedRectangle:
    return RoundedRectangle(
        corner_radius=radius,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=0.95,
        stroke_color=PANEL_2,
        stroke_width=1.5,
    )


def make_grid(spec: GridSpec, *, dot_radius: float = 0.055) -> tuple[VGroup, dict[tuple[int, int], Dot]]:
    dots: dict[tuple[int, int], Dot] = {}
    edges = VGroup()
    points = VGroup()
    w = (spec.cols - 1) * spec.spacing
    h = (spec.rows - 1) * spec.spacing
    x0, y0 = -w / 2, h / 2

    def pos(r: int, c: int):
        return [x0 + c * spec.spacing, y0 - r * spec.spacing, 0]

    for r in range(spec.rows):
        for c in range(spec.cols):
            d = Dot(pos(r, c), radius=dot_radius, color=GREY_B)
            dots[(r, c)] = d
            points.add(d)
            if c + 1 < spec.cols:
                edges.add(Line(pos(r, c), pos(r, c + 1), stroke_width=2, color=GREY_D))
            if r + 1 < spec.rows:
                edges.add(Line(pos(r, c), pos(r + 1, c), stroke_width=2, color=GREY_D))
    return VGroup(edges, points), dots


def path_lines(dots: dict[tuple[int, int], Dot], cells: list[tuple[int, int]], color=ACCENT, width: float = 6) -> VGroup:
    return VGroup(
        *[
            Line(dots[a].get_center(), dots[b].get_center(), stroke_width=width, color=color)
            for a, b in zip(cells, cells[1:])
        ]
    )


class OneesanGPUExplainer(Scene):
    """GPU 数え上げおねえさん問題の解説動画。

    Render:
        uv run manim -pqh src/manimations/oneesan_gpu.py OneesanGPUExplainer
    Fast preview:
        uv run manim -pql src/manimations/oneesan_gpu.py OneesanGPUExplainer
    """

    def construct(self) -> None:
        self.camera.background_color = BG
        self.title_scene()
        self.problem_scene()
        self.explosion_scene()
        self.frontier_scene()
        self.packing_scene()
        self.dense_scene()
        self.kernel_scene()
        self.benchmark_scene()
        self.multigpu_scene()
        self.outro_scene()

    def section_title(self, kicker: str, title: str, subtitle: str | None = None) -> VGroup:
        k = txt(kicker, 22, CYAN, bold=True)
        t = txt(title, 44, FG, bold=True)
        items = VGroup(k, t).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if subtitle:
            s = txt(subtitle, 23, MUTED)
            items.add(s)
            items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        items.to_edge(UP, buff=0.42).to_edge(LEFT, buff=0.55)
        return items

    def clear(self, run_time: float = 0.45) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)

    # ------------------------------------------------------------------
    # 0. Title
    # ------------------------------------------------------------------
    def title_scene(self) -> None:
        grid, dots = make_grid(GridSpec(6, 6, 0.58))
        grid.scale(0.92).shift(RIGHT * 3.45 + DOWN * 0.25)

        # grid was moved after dot construction; lines below use current centers.
        route = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1),
                 (2, 0), (3, 0), (4, 0), (4, 1), (3, 1), (3, 2), (4, 2), (5, 2),
                 (5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3), (0, 4), (1, 4),
                 (2, 4), (3, 4), (4, 4), (5, 4), (5, 5)]
        route_lines = path_lines(dots, route, ACCENT, 6)

        title = txt("GPUで救う\n数え上げおねえさん", 56, FG, bold=True)
        title.to_edge(LEFT, buff=0.68).shift(UP * 0.55)
        sub = txt("frontier DP × dense ranking × CUDA", 27, CYAN, mono=True)
        sub.next_to(title, DOWN, aligned_edge=LEFT, buff=0.35)
        tag = txt("2026 GPU implementation", 20, MUTED, mono=True)
        tag.next_to(sub, DOWN, aligned_edge=LEFT, buff=0.18)

        self.play(Write(title), FadeIn(sub, shift=UP * 0.15), FadeIn(tag), run_time=1.1)
        self.play(FadeIn(grid, shift=LEFT * 0.15), run_time=0.6)
        self.play(Create(route_lines), run_time=1.7)
        self.wait(0.7)
        self.clear()

    # ------------------------------------------------------------------
    # 1. Problem
    # ------------------------------------------------------------------
    def problem_scene(self) -> None:
        head = self.section_title("01 / PROBLEM", "同じ頂点を2度通らず、左上から右下へ", "上下左右には自由に進める。全部で何通り？")
        self.play(FadeIn(head, shift=DOWN * 0.15))

        grid, dots = make_grid(GridSpec(5, 5, 0.72))
        grid.shift(LEFT * 3.25 + DOWN * 0.55)
        start = Circle(radius=0.13, color=LIME, stroke_width=4).move_to(dots[(0, 0)])
        goal = Circle(radius=0.13, color=HOT, stroke_width=4).move_to(dots[(4, 4)])
        s_label = txt("START", 18, LIME, bold=True).next_to(start, LEFT, buff=0.18)
        g_label = txt("GOAL", 18, HOT, bold=True).next_to(goal, RIGHT, buff=0.18)

        self.play(Create(grid), FadeIn(start), FadeIn(goal), FadeIn(s_label), FadeIn(g_label), run_time=0.8)

        p1 = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2),
              (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (4, 1),
              (4, 2), (4, 3), (4, 4)]
        p2 = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3),
              (1, 3), (1, 2), (2, 2), (3, 2), (3, 3), (2, 3), (2, 4), (3, 4), (4, 4)]
        l1 = path_lines(dots, p1, ACCENT, 5)
        l2 = path_lines(dots, p2, PURPLE, 5)
        self.play(Create(l1), run_time=1.1)
        self.play(Transform(l1, l2), run_time=0.8)
        self.play(FadeOut(l1), run_time=0.3)

        rule_box = panel(5.25, 3.05).shift(RIGHT * 3.05 + DOWN * 0.45)
        rules = VGroup(
            txt("制約", 24, CYAN, bold=True),
            txt("• 上 / 下 / 左 / 右に1マス", 26),
            txt("• 一度通った頂点には戻れない", 26),
            txt("• START → GOAL の全自己回避路を数える", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(rule_box)
        self.play(FadeIn(rule_box), LaggedStart(*[FadeIn(x, shift=RIGHT * 0.1) for x in rules], lag_ratio=0.12), run_time=0.9)
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------
    # 2. Exponential explosion
    # ------------------------------------------------------------------
    def explosion_scene(self) -> None:
        head = self.section_title("02 / WHY HARD", "愚直探索は、分岐するたびに爆発する")
        self.play(FadeIn(head))

        root = Dot(LEFT * 5 + UP * 1.5, radius=0.08, color=FG)
        levels: list[VGroup] = []
        prev = VGroup(root)
        all_edges = VGroup()
        self.add(root)

        for depth in range(5):
            count = min(3 ** (depth + 1), 55)
            x = -3.3 + depth * 1.65
            ys = [2.0 - 4.0 * i / max(1, count - 1) for i in range(count)]
            nodes = VGroup(*[Dot([x, y, 0], radius=0.035, color=ACCENT) for y in ys])
            for i, node in enumerate(nodes):
                parent = prev[min(len(prev) - 1, i // 3)]
                all_edges.add(Line(parent, node, stroke_width=1.1, color=BLUE_E))
            self.play(Create(all_edges[-len(nodes):]), FadeIn(nodes), run_time=0.28)
            levels.append(nodes)
            prev = nodes

        big = txt("組合せ爆発", 46, HOT, bold=True).shift(RIGHT * 3.7 + UP * 0.55)
        small = txt("GPUで全探索しても\n枝ごとの制御フローがバラバラ", 26, MUTED).next_to(big, DOWN, buff=0.3)
        warp = VGroup(*[Square(0.13, stroke_width=0, fill_color=ACCENT, fill_opacity=0.9) for _ in range(32)])
        warp.arrange_in_grid(rows=2, cols=16, buff=0.04).next_to(small, DOWN, buff=0.5)
        for i, sq in enumerate(warp):
            if i % 4 != 0:
                sq.set_fill(GREY_D, opacity=0.28)
        warp_label = txt("warp: 多くの lane が待つ", 20, WARN, mono=True).next_to(warp, DOWN, buff=0.14)
        self.play(Write(big), FadeIn(small), FadeIn(warp), FadeIn(warp_label), run_time=0.8)
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------
    # 3. Frontier DP
    # ------------------------------------------------------------------
    def frontier_scene(self) -> None:
        head = self.section_title("03 / FRONTIER DP", "過去ぜんぶではなく、“境界”だけ覚える")
        self.play(FadeIn(head))

        grid, dots = make_grid(GridSpec(6, 7, 0.56))
        grid.shift(LEFT * 2.8 + DOWN * 0.45)
        self.play(FadeIn(grid), run_time=0.6)

        # processed region overlay
        x_left = dots[(2, 0)].get_center()[0] - 0.28
        x_front = dots[(2, 3)].get_center()[0] + 0.28
        y_top = dots[(0, 0)].get_center()[1] + 0.28
        y_bot = dots[(5, 0)].get_center()[1] - 0.28
        processed = Rectangle(
            width=x_front - x_left,
            height=y_top - y_bot,
            fill_color=BLUE_D,
            fill_opacity=0.22,
            stroke_width=0,
        ).move_to([(x_left + x_front) / 2, (y_top + y_bot) / 2, 0])
        frontier = DashedLine([x_front, y_top, 0], [x_front, y_bot, 0], color=CYAN, stroke_width=4)
        frontier_label = txt("frontier", 20, CYAN, mono=True).next_to(frontier, UP, buff=0.1)
        self.play(FadeIn(processed), Create(frontier), FadeIn(frontier_label))

        # Two very different histories that have same visible frontier summary.
        hist_a = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (3, 1), (3, 2)]
        hist_b = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2)]
        pa = path_lines(dots, hist_a, ACCENT, 4.5)
        pb = path_lines(dots, hist_b, PURPLE, 4.5)
        self.play(Create(pa), run_time=0.75)
        self.play(Transform(pa, pb), run_time=0.65)

        box = panel(4.8, 3.25).shift(RIGHT * 4.0 + DOWN * 0.35)
        t1 = txt("未来から見える情報が同じなら", 25, FG, bold=True)
        t2 = txt("それまでの経路は区別しなくていい", 25, FG)
        arrow = Arrow(LEFT, RIGHT, color=CYAN, buff=0).scale(0.65)
        collapse = txt("状態を1つに圧縮", 31, LIME, bold=True)
        content = VGroup(t1, t2, arrow, collapse).arrange(DOWN, buff=0.28).move_to(box)
        self.play(FadeIn(box), FadeIn(t1), FadeIn(t2), GrowArrow(arrow), Write(collapse), run_time=1.0)

        count_a = txt("1,000,000 本の途中経路", 22, MUTED, mono=True).next_to(box, DOWN, buff=0.25)
        count_b = txt("→ 1 frontier state + count", 22, CYAN, mono=True).next_to(count_a, DOWN, buff=0.09)
        self.play(FadeIn(count_a), FadeIn(count_b))
        self.wait(0.7)
        self.clear()

    # ------------------------------------------------------------------
    # 4. 2-bit packing
    # ------------------------------------------------------------------
    def packing_scene(self) -> None:
        head = self.section_title("04 / PACKED STATE", "frontier の各頂点を 2 bit で持つ", "MateID を packed integer に詰め込む")
        self.play(FadeIn(head))

        states = VGroup()
        labels = ["00", "01", "10", "11", "01", "00", "10", "01", "11", "00", "01", "10"]
        colors = [GREY_D, ACCENT, LIME, WARN]
        for i, bits in enumerate(labels):
            sq = RoundedRectangle(corner_radius=0.08, width=0.78, height=0.78,
                                  fill_color=PANEL_2, fill_opacity=1, stroke_color=colors[int(bits, 2)], stroke_width=2)
            b = txt(bits, 23, colors[int(bits, 2)], mono=True, bold=True).move_to(sq)
            idx = txt(str(i), 14, MUTED, mono=True).next_to(sq, DOWN, buff=0.06)
            states.add(VGroup(sq, b, idx))
        states.arrange(RIGHT, buff=0.11).scale(0.92).shift(UP * 0.45)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.12) for s in states], lag_ratio=0.05), run_time=0.8)

        brace = Brace(states, DOWN, color=CYAN)
        brace_text = txt("frontier state = 24 bit", 24, CYAN, mono=True).next_to(brace, DOWN, buff=0.12)
        self.play(GrowFromCenter(brace), FadeIn(brace_text))

        word = panel(9.5, 1.35, color="#11192b").shift(DOWN * 2.0)
        packed = txt("10 01 00 11 01 10 00 01 11 10 01 00", 28, LIME, mono=True, bold=True).move_to(word)
        arrow = Arrow(brace_text.get_bottom(), word.get_top(), color=CYAN, buff=0.18)
        self.play(GrowArrow(arrow), FadeIn(word), TransformFromCopy(VGroup(*[s[1] for s in states]), packed), run_time=0.9)

        foot = txt("小さい・比較しやすい・GPUレジスタで扱いやすい", 24, MUTED).next_to(word, DOWN, buff=0.28)
        self.play(FadeIn(foot))
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------
    # 5. Dense rank/unrank
    # ------------------------------------------------------------------
    def dense_scene(self) -> None:
        head = self.section_title("05 / DENSE RANKING", "hash table を捨てる", "有効状態 ↔ 連番 rank を直接変換")
        self.play(FadeIn(head))

        # Hash table side
        hash_box = panel(5.5, 4.4).shift(LEFT * 3.3 + DOWN * 0.55)
        h_title = txt("hash table", 28, HOT, mono=True, bold=True).next_to(hash_box.get_top(), DOWN, buff=0.25)
        buckets = VGroup()
        for i in range(6):
            b = Rectangle(width=4.5, height=0.45, stroke_color=GREY_D, fill_color=PANEL_2, fill_opacity=0.8)
            key = txt(f"0x{(i * 0x1D3 + 0x71):04x}", 18, MUTED, mono=True)
            val = txt(str([51, 18, 77, 9, 64, 33][i]), 18, FG, mono=True)
            key.move_to(b).shift(LEFT * 1.2)
            val.move_to(b).shift(RIGHT * 1.45)
            buckets.add(VGroup(b, key, val))
        buckets.arrange(DOWN, buff=0.07).move_to(hash_box).shift(DOWN * 0.2)
        random_arrows = VGroup(*[
            Arrow(LEFT * 5.5 + UP * (1.1 - i * 0.35), buckets[(i * 5) % 6].get_left(), buff=0.08, stroke_width=2, color=HOT)
            for i in range(6)
        ])

        # Dense array side
        dense_box = panel(5.5, 4.4).shift(RIGHT * 3.3 + DOWN * 0.55)
        d_title = txt("dense uint32[]", 28, LIME, mono=True, bold=True).next_to(dense_box.get_top(), DOWN, buff=0.25)
        cells = VGroup()
        for i in range(8):
            c = Rectangle(width=0.5, height=2.25, fill_color=BLUE_E, fill_opacity=0.75, stroke_color=ACCENT)
            idx = txt(str(i), 14, MUTED, mono=True).next_to(c, DOWN, buff=0.07)
            cells.add(VGroup(c, idx))
        cells.arrange(RIGHT, buff=0.04).move_to(dense_box).shift(DOWN * 0.08)
        stride = Arrow(cells[1].get_center(), cells[6].get_center(), buff=0.15, color=LIME)
        stride_label = txt("連続アクセス", 20, LIME, mono=True).next_to(stride, UP, buff=0.12)

        self.play(FadeIn(hash_box), FadeIn(dense_box), FadeIn(h_title), FadeIn(d_title))
        self.play(FadeIn(buckets), LaggedStart(*[GrowArrow(a) for a in random_arrows], lag_ratio=0.06), run_time=0.8)
        self.play(FadeIn(cells), GrowArrow(stride), FadeIn(stride_label), run_time=0.6)

        cross = txt("×", 58, HOT, bold=True).move_to(hash_box.get_corner(UP + RIGHT) + LEFT * 0.35 + DOWN * 0.35)
        check = txt("✓", 52, LIME, bold=True).move_to(dense_box.get_corner(UP + RIGHT) + LEFT * 0.4 + DOWN * 0.35)
        rank = txt("packed state  ⇄  rank", 25, CYAN, mono=True, bold=True).next_to(dense_box, DOWN, buff=0.24)
        self.play(FadeIn(cross), FadeIn(check), Write(rank))
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------
    # 6. Specialized CUDA kernel
    # ------------------------------------------------------------------
    def kernel_scene(self) -> None:
        head = self.section_title("06 / CUDA", "1遷移ずつ kernel launch しない", "transition-closed group を VRAM 上で 10–14 遷移まとめて処理")
        self.play(FadeIn(head))

        stages = VGroup()
        names = ["rank", "load", "transition × 12", "reduce", "store"]
        widths = [1.25, 1.25, 4.4, 1.25, 1.25]
        stage_colors = [CYAN, ACCENT, LIME, WARN, PURPLE]
        for name, width, color in zip(names, widths, stage_colors):
            r = RoundedRectangle(corner_radius=0.12, width=width, height=1.25,
                                 fill_color=color, fill_opacity=0.18, stroke_color=color, stroke_width=2.5)
            t = txt(name, 20 if "transition" not in name else 22, color, mono=True, bold=True).move_to(r)
            stages.add(VGroup(r, t))
        stages.arrange(RIGHT, buff=0.17).shift(UP * 0.3)
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.15) for s in stages], lag_ratio=0.09), run_time=0.8)

        # Data blocks flowing through the transition group.
        data = VGroup(*[
            Square(0.18, fill_color=ACCENT, fill_opacity=0.9, stroke_width=0) for _ in range(28)
        ]).arrange_in_grid(rows=4, cols=7, buff=0.045)
        data.move_to(stages[1]).shift(DOWN * 1.65)
        data_label = txt("state counts", 19, MUTED, mono=True).next_to(data, DOWN, buff=0.12)
        self.play(FadeIn(data), FadeIn(data_label))

        data_target = data.copy().move_to(stages[2]).shift(DOWN * 1.65)
        self.play(Transform(data, data_target), run_time=0.65)
        for i in range(3):
            self.play(
                AnimationGroup(*[Flash(s, color=LIME, flash_radius=0.23, line_length=0.07, run_time=0.25) for s in data[i * 7:(i + 1) * 7]], lag_ratio=0.03),
                run_time=0.3,
            )

        scratch = panel(4.3, 1.15, color="#102119").next_to(stages[2], DOWN, buff=1.1)
        scratch_t = txt("VRAM scratch stays resident", 20, LIME, mono=True).move_to(scratch)
        arrows = VGroup(
            Arrow(stages[2].get_bottom(), scratch.get_top(), color=LIME, buff=0.08),
            Arrow(scratch.get_top() + RIGHT * 1.2, stages[2].get_bottom() + RIGHT * 1.2, color=LIME, buff=0.08),
        )
        self.play(FadeIn(scratch), FadeIn(scratch_t), *[GrowArrow(a) for a in arrows])

        tagline = txt("launch overhead ↓    global-memory traffic ↓    locality ↑", 24, CYAN, mono=True, bold=True)
        tagline.to_edge(DOWN, buff=0.48)
        self.play(Write(tagline))
        self.wait(0.7)
        self.clear()

    # ------------------------------------------------------------------
    # 7. Benchmark
    # ------------------------------------------------------------------
    def benchmark_scene(self) -> None:
        head = self.section_title("07 / RTX 5090", "n=23 まで、単一GPUで実測", "specialized / HBM-resident 実装")
        self.play(FadeIn(head))

        data = [(18, 0.766), (20, 10.749), (21, 34.984), (22, 120.838), (23, 302.634)]
        max_t = max(t for _, t in data)
        bars = VGroup()
        labels = VGroup()
        values = VGroup()
        baseline_x = -4.65
        for i, (n, seconds) in enumerate(data):
            y = 1.45 - i * 0.72
            width = 7.3 * seconds / max_t
            bar = RoundedRectangle(corner_radius=0.06, width=max(0.08, width), height=0.42,
                                   fill_color=ACCENT if n < 23 else LIME,
                                   fill_opacity=0.88, stroke_width=0)
            bar.move_to([baseline_x + max(0.08, width) / 2, y, 0])
            label = txt(f"n={n}", 21, FG, mono=True, bold=True).move_to([baseline_x - 0.55, y, 0])
            value = txt(f"{seconds:.3f} s", 20, MUTED if n < 23 else LIME, mono=True, bold=n == 23)
            value.next_to(bar, RIGHT, buff=0.16)
            bars.add(bar)
            labels.add(label)
            values.add(value)

        self.play(LaggedStart(*[GrowFromPoint(b, b.get_left()) for b in bars], lag_ratio=0.08),
                  FadeIn(labels), run_time=1.1)
        self.play(LaggedStart(*[FadeIn(v, shift=RIGHT * 0.08) for v in values], lag_ratio=0.07), run_time=0.7)

        mem = panel(4.7, 1.55).to_edge(RIGHT, buff=0.55).shift(DOWN * 2.15)
        mem1 = txt("n=23 peak", 19, MUTED, mono=True)
        mem2 = txt("29.481 GiB", 32, WARN, mono=True, bold=True)
        mem3 = txt("32GB VRAM のほぼ限界", 19, WARN)
        VGroup(mem1, mem2, mem3).arrange(DOWN, buff=0.10).move_to(mem)
        self.play(FadeIn(mem), FadeIn(mem1), FadeIn(mem2), FadeIn(mem3))

        residue = txt("residue = 2762394459", 20, CYAN, mono=True).next_to(mem, DOWN, buff=0.18)
        self.play(FadeIn(residue))
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------
    # 8. Multi-GPU
    # ------------------------------------------------------------------
    def multigpu_scene(self) -> None:
        head = self.section_title("08 / MULTI GPU", "次は、状態空間そのものを分割する", "最終ターゲット: B300 × 8")
        self.play(FadeIn(head))

        gpu_row = VGroup()
        for i in range(8):
            box = RoundedRectangle(corner_radius=0.12, width=1.25, height=1.25,
                                   fill_color=BLUE_E, fill_opacity=0.65, stroke_color=ACCENT, stroke_width=2)
            t = txt(f"GPU {i}", 17, FG, mono=True, bold=True).move_to(box)
            gpu_row.add(VGroup(box, t))
        gpu_row.arrange(RIGHT, buff=0.17).shift(UP * 0.75)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.13) for g in gpu_row], lag_ratio=0.05), run_time=0.8)

        chunks = VGroup()
        for i in range(8):
            c = Rectangle(width=1.13, height=0.65, fill_color=[BLUE_D, PURPLE, GREEN_E, ORANGE][i % 4],
                          fill_opacity=0.7, stroke_width=0)
            t = txt(f"rank shard {i}", 13, FG, mono=True).move_to(c)
            chunks.add(VGroup(c, t))
        chunks.arrange(RIGHT, buff=0.29).next_to(gpu_row, DOWN, buff=0.55)
        self.play(LaggedStart(*[TransformFromCopy(gpu_row[i], chunks[i]) for i in range(8)], lag_ratio=0.06), run_time=0.8)

        fabric = Line(gpu_row[0].get_bottom() + DOWN * 1.65, gpu_row[-1].get_bottom() + DOWN * 1.65,
                      stroke_width=8, color=LIME)
        fabric_label = txt("NVLink / NVSwitch", 22, LIME, mono=True, bold=True).next_to(fabric, DOWN, buff=0.12)
        self.play(Create(fabric), FadeIn(fabric_label))

        # Cross-GPU traffic arrows: owner rank -> destination shard.
        arrows = VGroup()
        pairs = [(0, 5), (1, 3), (2, 7), (4, 0), (5, 6), (7, 2)]
        for a, b in pairs:
            arrows.add(Arrow(chunks[a].get_bottom(), chunks[b].get_bottom(), buff=0.12,
                             color=CYAN, stroke_width=2.5, max_tip_length_to_length_ratio=0.08))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.07), run_time=0.8)

        flow = VGroup(
            txt("state ownership", 19, MUTED, mono=True),
            txt("→ all-to-all", 19, MUTED, mono=True),
            txt("→ radix sort / reduce", 19, MUTED, mono=True),
            txt("→ compact", 19, LIME, mono=True, bold=True),
        ).arrange(RIGHT, buff=0.24).to_edge(DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(x, shift=RIGHT * 0.08) for x in flow], lag_ratio=0.1), run_time=0.8)
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------
    # 9. Outro
    # ------------------------------------------------------------------
    def outro_scene(self) -> None:
        one = txt("組合せ爆発を", 42, MUTED, bold=True)
        two = txt("GPUの物量だけで殴らない。", 50, FG, bold=True)
        three = txt("状態表現を変えてから、GPUに食わせる。", 42, LIME, bold=True)
        group = VGroup(one, two, three).arrange(DOWN, buff=0.28)
        self.play(FadeIn(one, shift=UP * 0.2))
        self.play(Write(two), run_time=0.8)
        self.play(Write(three), run_time=0.8)

        footer = txt("frontier DP  ×  packed MateID  ×  dense rank  ×  CUDA", 22, CYAN, mono=True)
        footer.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(footer))
        self.wait(1.2)
