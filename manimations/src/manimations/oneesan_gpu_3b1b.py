from __future__ import annotations

from dataclasses import dataclass

from manim import (
    Arrow,
    BLACK,
    BLUE_C,
    BLUE_D,
    BLUE_E,
    Brace,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_E,
    GrowArrow,
    GrowFromCenter,
    LaggedStart,
    LEFT,
    Line,
    MovingCameraScene,
    RIGHT,
    RoundedRectangle,
    Square,
    Text,
    Transform,
    TransformFromCopy,
    TransformMatchingShapes,
    UP,
    VGroup,
    WHITE,
    Write,
    config,
)

config.background_color = BLACK

SERIF = "DejaVu Serif"

FG = "#ECECF1"
MUTED = "#8A8F98"
GRID = "#343842"
BLUE1 = "#58C4DD"
GREEN1 = "#83C167"
YELLOW1 = "#FFFF00"
ORANGE1 = "#FF862F"
RED1 = "#FC6255"
PURPLE1 = "#9A72AC"


# -----------------------------------------------------------------------------
# Layout system
# -----------------------------------------------------------------------------

FRAME_W = float(config.frame_width)
FRAME_H = float(config.frame_height)
MARGIN_X = 0.52
MARGIN_Y = 0.34
SAFE_LEFT = -FRAME_W / 2 + MARGIN_X
SAFE_RIGHT = FRAME_W / 2 - MARGIN_X
SAFE_BOTTOM = -FRAME_H / 2 + MARGIN_Y
SAFE_TOP = FRAME_H / 2 - MARGIN_Y
SAFE_W = SAFE_RIGHT - SAFE_LEFT
SAFE_H = SAFE_TOP - SAFE_BOTTOM

TITLE_BAND_H = 1.02
FOOTER_BAND_H = 0.72
BODY_TOP = SAFE_TOP - TITLE_BAND_H
BODY_BOTTOM = SAFE_BOTTOM + FOOTER_BAND_H
BODY_H = BODY_TOP - BODY_BOTTOM


@dataclass(frozen=True)
class Rect:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.bottom + self.top) / 2


SAFE_RECT = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_TOP)
BODY_RECT = Rect(SAFE_LEFT, SAFE_RIGHT, BODY_BOTTOM, BODY_TOP)


def jtext(s: str, size: int = 34, color=FG, *, bold: bool = False, mono: bool = False) -> Text:
    return Text(
        s,
        font=SERIF,
        font_size=size,
        color=color,
        weight="BOLD" if bold else "NORMAL",
    )


def formula(s: str, size: int = 38, color=FG) -> Text:
    return Text(s, font=SERIF, font_size=size, color=color)


def flow_arrow(start, end, color=YELLOW1, *, buff: float = 0.06, stroke_width: float = 6.5):
    return Arrow(
        start=start,
        end=end,
        buff=buff,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.22,
        max_stroke_width_to_length_ratio=10,
    )


def thin_flow_arrow(start, end, color=GREEN1, *, buff: float = 0.05, stroke_width: float = 5.0):
    return Arrow(
        start=start,
        end=end,
        buff=buff,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.20,
        max_stroke_width_to_length_ratio=10,
    )


def comm_arrow(start, end, color=YELLOW1, *, angle: float = 0.35):
    return CurvedArrow(
        start_point=start,
        end_point=end,
        angle=angle,
        color=color,
        stroke_width=6.0,
        tip_length=0.22,
    )


def linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [(a + b) / 2]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def fit_to_box(mob, *, max_width: float | None = None, max_height: float | None = None):
    """Scale down, never up, so mob fits the requested bounding box."""
    scales = [1.0]
    if max_width is not None and mob.width > 0:
        scales.append(max_width / mob.width)
    if max_height is not None and mob.height > 0:
        scales.append(max_height / mob.height)
    scale = min(scales)
    if scale < 1.0:
        mob.scale(scale)
    return mob


def center_in(mob, rect: Rect, *, pad: float = 0.0):
    fit_to_box(
        mob,
        max_width=max(0.01, rect.width - 2 * pad),
        max_height=max(0.01, rect.height - 2 * pad),
    )
    mob.move_to([rect.cx, rect.cy, 0])
    return mob


def place_top_center(mob, *, top: float = SAFE_TOP, max_width: float = SAFE_W):
    fit_to_box(mob, max_width=max_width)
    mob.move_to([0, top - mob.height / 2, 0])
    return mob


def arrange_row_exact(
    mobs,
    *,
    left: float,
    right: float,
    y: float,
    min_gap: float = 0.08,
    max_gap: float = 0.32,
):
    """Lay objects out from their actual widths; shrink uniformly if required."""
    items = list(mobs)
    if not items:
        return VGroup()

    available = right - left
    total_width = sum(m.width for m in items)
    required = total_width + min_gap * (len(items) - 1)
    if required > available and total_width > 0:
        scale = (available - min_gap * (len(items) - 1)) / total_width
        scale = max(0.05, scale)
        for mob in items:
            mob.scale(scale)
        total_width = sum(m.width for m in items)

    if len(items) == 1:
        gap = 0.0
    else:
        gap = min(max_gap, max(min_gap, (available - total_width) / (len(items) - 1)))

    occupied = total_width + gap * (len(items) - 1)
    x = (left + right - occupied) / 2
    for mob in items:
        mob.move_to([x + mob.width / 2, y, 0])
        x += mob.width + gap
    return VGroup(*items)


def bounds(mob) -> tuple[float, float, float, float]:
    return (
        float(mob.get_left()[0]),
        float(mob.get_right()[0]),
        float(mob.get_bottom()[1]),
        float(mob.get_top()[1]),
    )


def overlaps(a, b, *, pad: float = 0.0) -> bool:
    al, ar, ab, at = bounds(a)
    bl, br, bb, bt = bounds(b)
    return not (
        ar + pad <= bl
        or br + pad <= al
        or at + pad <= bb
        or bt + pad <= ab
    )


def make_lattice(rows: int, cols: int, spacing: float = 0.62):
    dots: dict[tuple[int, int], Circle] = {}
    lines = VGroup()
    pts = VGroup()
    x0 = -(cols - 1) * spacing / 2
    y0 = +(rows - 1) * spacing / 2

    def p(r: int, c: int):
        return [x0 + c * spacing, y0 - r * spacing, 0]

    for r in range(rows):
        for c in range(cols):
            d = Circle(radius=0.032, stroke_width=0, fill_color=WHITE, fill_opacity=0.85).move_to(p(r, c))
            dots[(r, c)] = d
            pts.add(d)
            if c + 1 < cols:
                lines.add(Line(p(r, c), p(r, c + 1), stroke_width=1.5, color=GRID))
            if r + 1 < rows:
                lines.add(Line(p(r, c), p(r + 1, c), stroke_width=1.5, color=GRID))
    return VGroup(lines, pts), dots


def path_from(dots, cells, color=BLUE1, width=5.0):
    return VGroup(*[
        Line(dots[a].get_center(), dots[b].get_center(), stroke_width=width, color=color)
        for a, b in zip(cells, cells[1:])
    ])


class OneesanGPU3B1B(MovingCameraScene):
    """3b1b-inspired GPU counting explainer with measured layouts."""

    def wipe(self, run_time: float = 0.42):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=run_time)

    def title(self, text: str, *, size: int = 40, color=FG):
        mob = jtext(text, size, color, bold=True)
        return place_top_center(mob, top=SAFE_TOP, max_width=SAFE_W)

    def audit(self, scene: str, **named_mobs):
        eps = 1e-5
        for name, mob in named_mobs.items():
            l, r, b, t = bounds(mob)
            print(f"[layout] {scene:12s} {name:18s} x=[{l:6.3f},{r:6.3f}] y=[{b:6.3f},{t:6.3f}] w={r-l:5.3f} h={t-b:5.3f}")
            if l < SAFE_LEFT - eps or r > SAFE_RIGHT + eps or b < SAFE_BOTTOM - eps or t > SAFE_TOP + eps:
                raise ValueError(
                    f"layout overflow in {scene}/{name}: "
                    f"bounds={(l, r, b, t)}, safe={(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_TOP)}"
                )

    def audit_no_overlap(self, scene: str, a_name: str, a, b_name: str, b, *, pad: float = 0.04):
        if overlaps(a, b, pad=pad):
            raise ValueError(f"layout overlap in {scene}: {a_name} vs {b_name}")

    def construct(self):
        self.camera.background_color = BLACK
        self.intro()
        self.search_explosion()
        self.frontier_idea()
        self.state_merging()
        self.packed_state()
        self.dense_ranking()
        self.cuda_grouping()
        self.benchmark()
        self.multigpu()
        self.outro()

    # ------------------------------------------------------------------
    # 1. Problem
    # ------------------------------------------------------------------
    def intro(self):
        title = self.title("Counting Oneesan", size=48)
        subtitle = jtext("Count every self-avoiding path on the grid", 27, MUTED)
        subtitle.next_to(title, DOWN, buff=0.12)
        fit_to_box(subtitle, max_width=SAFE_W)

        content_top = subtitle.get_bottom()[1] - 0.18
        content_rect = Rect(SAFE_LEFT, SAFE_RIGHT, BODY_BOTTOM + 0.42, content_top)

        grid, dots = make_lattice(7, 7, 0.55)
        center_in(grid, content_rect, pad=0.35)

        start = Circle(radius=0.11, color=GREEN1, stroke_width=3).move_to(dots[(0, 0)])
        goal = Circle(radius=0.11, color=RED1, stroke_width=3).move_to(dots[(6, 6)])

        route = [
            (0,0),(0,1),(1,1),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,3),
            (1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(4,0),(5,0),(5,1),(4,1),
            (4,2),(5,2),(6,2),(6,3),(5,3),(4,3),(4,4),(3,4),(2,4),(1,4),
            (0,4),(0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(5,6),(6,6),
        ]
        path = path_from(dots, route, BLUE1, 5.2)

        count = formula("Nₙ = # { self-avoiding paths }", 35, FG)
        fit_to_box(count, max_width=SAFE_W)
        count.move_to([0, SAFE_BOTTOM + count.height / 2, 0])

        self.audit("intro", title=title, subtitle=subtitle, grid=grid, count=count)
        self.audit_no_overlap("intro", "subtitle", subtitle, "grid", grid)
        self.audit_no_overlap("intro", "grid", grid, "count", count)

        self.play(FadeIn(grid, scale=0.96), FadeIn(start), FadeIn(goal), run_time=0.75)
        self.play(Create(path), run_time=2.1)
        self.play(FadeIn(title, shift=UP * 0.10), FadeIn(subtitle, shift=UP * 0.10), run_time=0.65)
        self.play(Write(count), run_time=0.75)
        self.wait(0.45)
        self.wipe()

    # ------------------------------------------------------------------
    # 2. Search explosion
    # ------------------------------------------------------------------
    def search_explosion(self):
        title = self.title("Why not enumerate every path?", size=40)

        split_x = SAFE_LEFT + SAFE_W * 0.62
        tree_rect = Rect(SAFE_LEFT, split_x - 0.28, BODY_BOTTOM, BODY_TOP)
        text_rect = Rect(split_x + 0.28, SAFE_RIGHT, BODY_BOTTOM, BODY_TOP)

        depth_x = linspace(tree_rect.left + 0.25, tree_rect.right - 0.18, 6)
        root = Circle(radius=0.07, fill_color=WHITE, fill_opacity=1, stroke_width=0).move_to([depth_x[0], tree_rect.cy, 0])
        prev = [root]
        tree = VGroup(root)
        levels: list[VGroup] = []

        for depth in range(5):
            n = min(3 ** (depth + 1), 81)
            ys = linspace(tree_rect.top - 0.18, tree_rect.bottom + 0.18, n)
            nodes = VGroup()
            edges = VGroup()
            for i, y in enumerate(ys):
                node = Circle(
                    radius=0.034 if depth < 3 else 0.021,
                    fill_color=BLUE1,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([depth_x[depth + 1], y, 0])
                parent = prev[min(len(prev) - 1, i // 3)]
                edges.add(Line(parent.get_center(), node.get_center(), stroke_width=1.15, color=BLUE_E))
                nodes.add(node)
            tree.add(edges, nodes)
            levels.append(VGroup(edges, nodes))
            prev = list(nodes)

        exp = formula("Θ(cⁿ²)", 64, RED1)
        bad = jtext("Storing the entire path as state\ngrows far too quickly", 25, MUTED)
        annotation = VGroup(exp, bad).arrange(DOWN, buff=0.28)
        center_in(annotation, text_rect, pad=0.20)

        self.audit("search", title=title, tree=tree, annotation=annotation)
        self.audit_no_overlap("search", "tree", tree, "annotation", annotation, pad=0.10)

        self.play(Write(title))
        self.play(FadeIn(root))
        for level in levels:
            edges, nodes = level
            self.play(Create(edges), FadeIn(nodes), run_time=0.38)
        self.play(Write(exp), FadeIn(bad, shift=UP * 0.08))

        target = VGroup(*[
            Circle(radius=0.05, fill_color=BLUE1, fill_opacity=1, stroke_width=0)
            for _ in range(5)
        ])
        target.arrange(DOWN, buff=0.58)
        center_in(target, Rect(tree_rect.cx - 0.35, tree_rect.cx + 0.35, tree_rect.bottom, tree_rect.top))
        self.camera.frame.save_state()
        self.play(
            FadeOut(title), FadeOut(annotation),
            Transform(tree, target),
            self.camera.frame.animate.set(width=FRAME_W * 0.62).move_to(target),
            run_time=0.95,
        )
        self.wait(0.18)
        self.play(self.camera.frame.animate.restore(), FadeOut(tree), run_time=0.45)

    # ------------------------------------------------------------------
    # 3. Frontier DP
    # ------------------------------------------------------------------
    def frontier_idea(self):
        title_a = self.title("The future only needs", size=31)
        title_b = jtext("the frontier, not the whole history", 38, BLUE1, bold=True)
        title_b.next_to(title_a, DOWN, buff=0.08)
        fit_to_box(title_b, max_width=SAFE_W)

        grid_top = title_b.get_bottom()[1] - 0.18
        state_h = 0.52
        state = formula("s = boundary connectivity", 33, FG)
        fit_to_box(state, max_width=SAFE_W)
        state.move_to([0, SAFE_BOTTOM + state.height / 2, 0])
        grid_rect = Rect(SAFE_LEFT, SAFE_RIGHT, state.get_top()[1] + 0.22, grid_top)

        grid, dots = make_lattice(6, 9, 0.58)
        center_in(grid, grid_rect, pad=0.36)

        cut_x = (dots[(0, 2)].get_center()[0] + dots[(0, 3)].get_center()[0]) / 2
        y_top = dots[(0, 0)].get_center()[1] + 0.28
        y_bot = dots[(5, 0)].get_center()[1] - 0.28
        cut = DashedLine([cut_x, y_top, 0], [cut_x, y_bot, 0], color=YELLOW1, stroke_width=4)
        frontier_label = formula("frontier", 27, YELLOW1).next_to(cut, UP, buff=0.07)

        left_history = [
            (0,0),(1,0),(1,1),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),
            (3,0),(3,1),(4,1),(4,0),(5,0),(5,1),(5,2),(4,2),(3,2),
        ]
        path = path_from(dots, left_history, BLUE1, 4.5)

        left_part = VGroup(*[dots[(r, c)] for r in range(6) for c in range(3)])
        frontier_nodes = VGroup(*[dots[(r, 3)] for r in range(6)])

        self.audit("frontier", title_a=title_a, title_b=title_b, grid=grid, state=state)
        self.audit_no_overlap("frontier", "title_b", title_b, "grid", grid)
        self.audit_no_overlap("frontier", "grid", grid, "state", state)

        self.play(FadeIn(grid), run_time=0.55)
        self.play(Create(path), run_time=1.05)
        self.play(Create(cut), FadeIn(frontier_label), run_time=0.50)
        self.play(FadeIn(title_a), Write(title_b))
        self.play(
            path.animate.set_opacity(0.15),
            left_part.animate.set_opacity(0.15),
            frontier_nodes.animate.set_color(YELLOW1).scale(1.65),
            run_time=0.72,
        )
        self.play(Write(state))

        next_cut_x = (dots[(0, 5)].get_center()[0] + dots[(0, 6)].get_center()[0]) / 2
        self.play(cut.animate.shift(RIGHT * (next_cut_x - cut_x)), frontier_label.animate.shift(RIGHT * (next_cut_x - cut_x)), run_time=1.0)
        self.wait(0.4)
        self.wipe()

    # ------------------------------------------------------------------
    # 4. State merging
    # ------------------------------------------------------------------
    def state_merging(self):
        title = self.title("Different histories merge when their frontier state matches", size=34)

        usable_top = title.get_bottom()[1] - 0.20
        footer_h = 0.70
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + footer_h)
        body = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.12, usable_top)

        w = body.width
        history_rect = Rect(body.left, body.left + 0.43 * w, body.bottom, body.top)
        state_rect = Rect(body.left + 0.50 * w, body.left + 0.66 * w, body.bottom, body.top)
        sum_rect = Rect(body.left + 0.74 * w, body.right, body.bottom, body.top)

        histories = VGroup()
        routes = [
            [(0,0),(1,0),(1,1),(0,1),(0,2),(1,2),(2,2),(2,3)],
            [(0,0),(0,1),(1,1),(1,0),(2,0),(2,1),(2,2),(2,3)],
            [(0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(1,2),(2,2),(2,3)],
            [(0,0),(0,1),(0,2),(1,2),(1,1),(2,1),(2,2),(2,3)],
            [(0,0),(1,0),(1,1),(2,1),(2,0),(3,0),(3,1),(3,2),(2,2),(2,3)],
            [(0,0),(0,1),(1,1),(2,1),(2,2),(1,2),(1,3),(2,3)],
        ]
        colors = [BLUE1, PURPLE1, GREEN1, ORANGE1, BLUE_C, RED1]
        for route, color in zip(routes, colors):
            g, dots = make_lattice(4, 4, 0.22)
            p = path_from(dots, route, color, 2.2)
            histories.add(VGroup(g, p))
        histories.arrange_in_grid(rows=2, cols=3, buff=(0.34, 0.34))
        center_in(histories, history_rect, pad=0.10)

        state_nodes = VGroup(*[
            Circle(radius=0.11, stroke_width=3, color=color, fill_opacity=0.08)
            for color in [YELLOW1, BLUE1, BLUE1, YELLOW1]
        ])
        state_nodes.arrange(DOWN, buff=0.44)
        links = VGroup(
            Line(state_nodes[0].get_center(), state_nodes[3].get_center(), color=YELLOW1, stroke_width=4),
            Line(state_nodes[1].get_center(), state_nodes[2].get_center(), color=BLUE1, stroke_width=4),
        )
        state_group = VGroup(links, state_nodes)
        center_in(state_group, state_rect, pad=0.12)

        sigma = formula("Σ", 76, GREEN1)
        eq = formula("dp[s] += count", 34, FG)
        sum_group = VGroup(sigma, eq).arrange(DOWN, buff=0.18)
        center_in(sum_group, sum_rect, pad=0.10)

        merge_targets = linspace(state_group.get_top()[1] - 0.16, state_group.get_bottom()[1] + 0.16, len(histories))
        arrows = VGroup(*[
            thin_flow_arrow(
                h.get_right(),
                [state_group.get_left()[0], merge_targets[i], 0],
                color=BLUE1 if i % 2 == 0 else GREEN1,
                buff=0.06,
                stroke_width=4.8,
            )
            for i, h in enumerate(histories)
        ])

        one = formula("10⁶ histories", 30, MUTED)
        two = formula("→ 1 state + 1 counter", 30, GREEN1)
        footer = arrange_row_exact([one, two], left=footer_rect.left, right=footer_rect.right, y=footer_rect.cy, min_gap=0.28, max_gap=0.55)

        self.audit("merge", title=title, histories=histories, state=state_group, sum=sum_group, footer=footer)
        self.audit_no_overlap("merge", "histories", histories, "state", state_group, pad=0.10)
        self.audit_no_overlap("merge", "state", state_group, "sum", sum_group, pad=0.10)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(h, scale=0.95) for h in histories], lag_ratio=0.06), run_time=0.78)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.04), run_time=0.68)
        self.play(FadeIn(state_group, scale=0.85), run_time=0.52)
        self.play(TransformFromCopy(state_group, sigma), Write(eq))
        self.play(FadeIn(one), Write(two))
        self.wait(0.45)
        self.wipe()

    # ------------------------------------------------------------------
    # 5. Packed state
    # ------------------------------------------------------------------
    def packed_state(self):
        title = self.title("Now compress the frontier state into an integer", size=38)
        body_top = title.get_bottom()[1] - 0.20
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + 0.76)
        body = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.12, body_top)

        left_rect = Rect(body.left, body.left + body.width * 0.42, body.bottom, body.top)
        right_rect = Rect(body.left + body.width * 0.50, body.right, body.bottom, body.top)

        nodes = VGroup(*[
            Circle(radius=0.13, stroke_width=3, color=color, fill_opacity=0.06)
            for color in [MUTED, BLUE1, GREEN1, YELLOW1, BLUE1, MUTED]
        ])
        nodes.arrange(DOWN, buff=0.34)
        labels = VGroup(*[
            formula(bits, 27, color)
            for bits, color in zip(["00", "01", "10", "11", "01", "00"], [MUTED, BLUE1, GREEN1, YELLOW1, BLUE1, MUTED])
        ])
        labels.arrange(DOWN, buff=0.30)
        state_list = VGroup(nodes, labels).arrange(RIGHT, buff=0.30)
        center_in(state_list, left_rect, pad=0.18)

        bits = formula("00  01  10  11  01  00", 44, FG)
        fit_to_box(bits, max_width=right_rect.width - 0.24)
        brace = Brace(bits, DOWN, color=WHITE)
        btxt = formula("2w bits", 27, MUTED).next_to(brace, DOWN, buff=0.08)
        bit_group = VGroup(bits, brace, btxt)
        center_in(bit_group, right_rect, pad=0.20)

        packed = formula("uint64_t state", 37, GREEN1)
        center_in(packed, footer_rect, pad=0.05)

        arrow = Arrow(bit_group.get_bottom(), packed.get_top(), color=GREEN1, buff=0.10)

        self.audit("packed", title=title, state_list=state_list, bits=bit_group, packed=packed)
        self.audit_no_overlap("packed", "state_list", state_list, "bits", bit_group, pad=0.14)

        self.play(Write(title))
        self.play(FadeIn(nodes))
        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.06) for x in labels], lag_ratio=0.07), run_time=0.70)
        self.play(TransformMatchingShapes(labels.copy(), bits), run_time=0.90)
        self.play(GrowFromCenter(brace), FadeIn(btxt))
        self.play(GrowArrow(arrow), TransformFromCopy(bits, packed), run_time=0.82)
        self.wait(0.40)
        self.wipe()

    # ------------------------------------------------------------------
    # 6. Dense ranking
    # ------------------------------------------------------------------
    def dense_ranking(self):
        title = self.title("On a GPU, dense arrays beat hash tables", size=36)
        body_top = title.get_bottom()[1] - 0.18
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + 0.72)
        body = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.15, body_top)

        left_rect = Rect(body.left, body.left + 0.34 * body.width, body.bottom, body.top)
        array_rect = Rect(body.left + 0.40 * body.width, body.right, body.bottom, body.top)

        state = formula("s", 54, YELLOW1)
        rank = formula("r = rank(s)", 40, BLUE1)
        sr = arrange_row_exact([state, rank], left=left_rect.left, right=left_rect.right, y=left_rect.cy + 0.38, min_gap=0.60, max_gap=0.90)
        sr_arrow = flow_arrow(state.get_right(), rank.get_left(), color=YELLOW1, buff=0.10, stroke_width=6.5)

        cells = VGroup(*[
            Square(0.32, stroke_width=1.1, stroke_color=GRID, fill_color=BLUE_E, fill_opacity=0.45)
            for _ in range(24)
        ])
        arrange_row_exact(cells, left=array_rect.left, right=array_rect.right, y=array_rect.cy + 0.52, min_gap=0.015, max_gap=0.035)

        pointer = flow_arrow(rank.get_right(), cells[11].get_top(), color=BLUE1, buff=0.08, stroke_width=6.0)
        idx = formula("dp[r]", 31, BLUE1).next_to(cells[11], DOWN, buff=0.20)

        lanes = VGroup(*[
            RoundedRectangle(corner_radius=0.03, width=0.23, height=0.23, stroke_width=0, fill_color=GREEN1, fill_opacity=1)
            for _ in range(8)
        ])
        arrange_row_exact(
            lanes,
            left=cells[8].get_left()[0],
            right=cells[15].get_right()[0],
            y=array_rect.cy - 0.72,
            min_gap=0.05,
            max_gap=0.13,
        )
        lane_arrows = VGroup(*[
            thin_flow_arrow(
                lane.get_top(),
                cells[8 + i].get_bottom(),
                color=GREEN1,
                buff=0.04,
                stroke_width=4.8,
            )
            for i, lane in enumerate(lanes)
        ])
        co = formula("coalesced", 24, GREEN1).next_to(lanes, DOWN, buff=0.12)

        old = formula("hash probe  →  random memory", 27, RED1)
        new = formula("rank  →  dense array", 27, GREEN1)
        center_in(old, footer_rect, pad=0.06)
        new.move_to(old)

        self.audit("dense", title=title, sr=sr, cells=cells, lanes=lanes, footer=old)
        self.audit_no_overlap("dense", "sr", sr, "cells", cells, pad=0.18)
        self.audit_no_overlap("dense", "cells", cells, "lanes", lanes, pad=0.12)

        self.play(FadeIn(title))
        self.play(Write(state))
        self.play(GrowArrow(sr_arrow), Write(rank))
        self.play(FadeIn(cells), run_time=0.62)
        self.play(GrowArrow(pointer), FadeIn(idx))
        self.play(FadeIn(lanes), LaggedStart(*[GrowArrow(a) for a in lane_arrows], lag_ratio=0.04), run_time=0.66)
        self.play(FadeIn(co))
        self.play(Write(old))
        self.play(TransformMatchingShapes(old, new), run_time=0.72)
        self.wait(0.42)
        self.wipe()

    # ------------------------------------------------------------------
    # 7. CUDA grouping
    # ------------------------------------------------------------------
    def cuda_grouping(self):
        title = self.title("Do not launch one kernel per transition", size=40)
        body_top = title.get_bottom()[1] - 0.18
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + 0.72)
        body = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.15, body_top)

        kernels = VGroup(*[
            RoundedRectangle(corner_radius=0.05, width=0.65, height=0.42, stroke_width=2, stroke_color=BLUE1)
            for _ in range(12)
        ])
        arrange_row_exact(kernels, left=body.left + 0.24, right=body.right - 0.24, y=body.top - 0.70, min_gap=0.08, max_gap=0.18)
        labels = VGroup(*[
            formula(str(i + 1), 17, BLUE1).move_to(k)
            for i, k in enumerate(kernels)
        ])

        arrow_len = min(0.52, body.height * 0.12)
        overhead_idx = [1, 4, 7, 10]
        overhead = VGroup(*[
            flow_arrow(
                kernels[i].get_bottom(),
                kernels[i].get_bottom() + DOWN * arrow_len,
                color=RED1,
                buff=0.02,
                stroke_width=6.0,
            )
            for i in overhead_idx
        ])
        oh = formula("launch / reload / store", 23, RED1)
        oh.next_to(overhead, DOWN, buff=0.10)

        fused_width = min(body.width * 0.70, SAFE_W - 0.8)
        fused = RoundedRectangle(
            corner_radius=0.12,
            width=fused_width,
            height=min(1.30, body.height * 0.30),
            stroke_width=3,
            stroke_color=GREEN1,
            fill_color=GREEN_E,
            fill_opacity=0.12,
        ).move_to([body.cx, body.cy + 0.10, 0])
        fused_txt = formula("transition-closed group   (10–14 transitions)", 29, GREEN1)
        fit_to_box(fused_txt, max_width=fused.width - 0.34)
        fused_txt.move_to(fused)

        vram = formula("VRAM-resident scratch", 29, YELLOW1)
        vram.move_to([body.cx, body.bottom + 0.44, 0])
        loop1 = flow_arrow(
            fused.get_bottom() + LEFT * fused.width * 0.22,
            vram.get_top() + LEFT * vram.width * 0.22,
            color=YELLOW1,
            buff=0.08,
            stroke_width=6.0,
        )
        loop2 = flow_arrow(
            vram.get_top() + RIGHT * vram.width * 0.22,
            fused.get_bottom() + RIGHT * fused.width * 0.22,
            color=YELLOW1,
            buff=0.08,
            stroke_width=6.0,
        )

        good = formula("less traffic   +   less launch overhead", 29, GREEN1)
        center_in(good, footer_rect, pad=0.06)

        self.audit("cuda", title=title, kernels=kernels, overhead=overhead, fused=fused, vram=vram, footer=good)
        self.audit_no_overlap("cuda", "fused", fused, "vram", vram, pad=0.10)

        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(VGroup(k, l), shift=UP * 0.08) for k, l in zip(kernels, labels)], lag_ratio=0.05), run_time=0.70)
        self.play(LaggedStart(*[GrowArrow(a) for a in overhead], lag_ratio=0.025), run_time=0.52)
        self.play(FadeIn(oh))
        self.play(FadeOut(overhead), FadeOut(oh), Transform(kernels, fused), FadeOut(labels), FadeIn(fused_txt), run_time=0.88)
        self.play(FadeIn(vram), GrowArrow(loop1), GrowArrow(loop2))
        self.play(Write(good))
        self.wait(0.48)
        self.wipe()

    # ------------------------------------------------------------------
    # 8. Benchmark
    # ------------------------------------------------------------------
    def benchmark(self):
        title = self.title("RTX 5090     n=23", size=43)
        body_top = title.get_bottom()[1] - 0.18
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + 0.78)
        plot_rect = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.15, body_top)

        data = [(18, 0.766), (20, 10.749), (21, 34.984), (22, 120.838), (23, 302.634)]
        labels = [formula(f"n={n}", 27, FG) for n, _ in data]
        vals = [formula(f"{t:.3f} s", 25, YELLOW1 if n == 23 else MUTED) for n, t in data]
        max_label_w = max(x.width for x in labels)
        max_val_w = max(x.width for x in vals)
        label_gap = 0.22
        value_gap = 0.18
        bar_left = plot_rect.left + max_label_w + label_gap
        bar_right = plot_rect.right - max_val_w - value_gap
        bar_max_w = bar_right - bar_left
        if bar_max_w <= 1.0:
            raise ValueError("benchmark plot has insufficient horizontal space")

        ys = linspace(plot_rect.top - 0.34, plot_rect.bottom + 0.34, len(data))
        bars = VGroup()
        label_group = VGroup()
        value_group = VGroup()
        max_t = max(t for _, t in data)

        for (n, t), y, lab, val in zip(data, ys, labels, vals):
            width = max(0.10, bar_max_w * (t / max_t) ** 0.55)
            line = Line([bar_left, y, 0], [bar_left + width, y, 0], stroke_width=13, color=YELLOW1 if n == 23 else BLUE1)
            dot = Circle(radius=0.065, stroke_width=0, fill_color=line.color, fill_opacity=1).move_to(line.get_end())
            lab.move_to([bar_left - label_gap - lab.width / 2, y, 0])
            val.move_to([bar_right + value_gap + val.width / 2, y, 0])
            bars.add(VGroup(line, dot))
            label_group.add(lab)
            value_group.add(val)

        mem = formula("peak VRAM   29.481 GiB", 34, ORANGE1)
        center_in(mem, footer_rect, pad=0.06)

        all_plot = VGroup(label_group, bars, value_group)
        self.audit("bench", title=title, plot=all_plot, footer=mem)
        self.audit_no_overlap("bench", "plot", all_plot, "footer", mem, pad=0.10)

        self.play(Write(title))
        self.play(FadeIn(label_group))
        self.play(LaggedStart(*[Create(b[0]) for b in bars], lag_ratio=0.07), run_time=0.95)
        self.play(LaggedStart(*[FadeIn(VGroup(b[1], v)) for b, v in zip(bars, value_group)], lag_ratio=0.05), run_time=0.62)
        self.play(Write(mem))
        self.wait(0.48)
        self.wipe()

    # ------------------------------------------------------------------
    # 9. Multi-GPU
    # ------------------------------------------------------------------
    def multigpu(self):
        title = self.title("Finally, partition the state space itself", size=38)
        body_top = title.get_bottom()[1] - 0.20
        footer_rect = Rect(SAFE_LEFT, SAFE_RIGHT, SAFE_BOTTOM, SAFE_BOTTOM + 0.72)
        body = Rect(SAFE_LEFT, SAFE_RIGHT, footer_rect.top + 0.16, body_top)

        colors = [BLUE1, PURPLE1, GREEN1, ORANGE1, BLUE_C, RED1, YELLOW1, BLUE_D]
        n = 8
        shard_gap = 0.035
        shard_w = (body.width - shard_gap * (n - 1)) / n
        shard_y = body.top - 0.70
        shards = VGroup(*[
            RoundedRectangle(corner_radius=0.035, width=shard_w, height=0.40, stroke_width=0, fill_color=colors[i], fill_opacity=0.88)
            for i in range(n)
        ])
        arrange_row_exact(shards, left=body.left, right=body.right, y=shard_y, min_gap=shard_gap, max_gap=shard_gap)

        number_line = Line([body.left, shard_y, 0], [body.right, shard_y, 0], color=WHITE, stroke_width=2.0)
        ticks = VGroup(*[
            Line([x, shard_y - 0.22, 0], [x, shard_y + 0.22, 0], color=WHITE, stroke_width=1.4)
            for x in linspace(body.left, body.right, n + 1)
        ])

        gpu_gap = 0.16
        gpu_w = min(1.0, (body.width - gpu_gap * (n - 1)) / n)
        gpu_y = body.cy - 0.30
        gpus = VGroup()
        for i in range(n):
            box = RoundedRectangle(corner_radius=0.08, width=gpu_w, height=0.78, stroke_width=2, stroke_color=colors[i], fill_opacity=0.03)
            label = formula(f"G{i}", 22, colors[i]).move_to(box)
            gpus.add(VGroup(box, label))
        arrange_row_exact(gpus, left=body.left, right=body.right, y=gpu_y, min_gap=gpu_gap, max_gap=gpu_gap)

        fabric_y = gpus.get_bottom()[1] - 0.58
        fabric = Line([gpus.get_left()[0], fabric_y, 0], [gpus.get_right()[0], fabric_y, 0], color=GREEN1, stroke_width=5)
        fabtxt = formula("NVLink / NVSwitch", 27, GREEN1).next_to(fabric, DOWN, buff=0.10)

        msgs = VGroup(
            comm_arrow(gpus[0].get_bottom(), gpus[5].get_bottom(), color=YELLOW1, angle=0.28),
            comm_arrow(gpus[2].get_bottom(), gpus[7].get_bottom(), color=YELLOW1, angle=0.32),
            comm_arrow(gpus[6].get_bottom(), gpus[1].get_bottom(), color=ORANGE1, angle=-0.30),
            comm_arrow(gpus[4].get_bottom(), gpus[3].get_bottom(), color=ORANGE1, angle=-0.22),
        )

        target = formula("B300 × 8", 43, YELLOW1)
        center_in(target, footer_rect, pad=0.04)

        self.audit("multigpu", title=title, shards=shards, gpus=gpus, fabric=VGroup(fabric, fabtxt), target=target)
        self.audit_no_overlap("multigpu", "gpus", gpus, "fabric", VGroup(fabric, fabtxt), pad=0.08)

        self.play(Write(title))
        self.play(Create(number_line), Create(ticks))
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.06) for s in shards], lag_ratio=0.04), run_time=0.62)
        self.play(LaggedStart(*[TransformFromCopy(shards[i], gpus[i]) for i in range(n)], lag_ratio=0.04), run_time=0.70)
        self.play(Create(fabric), FadeIn(fabtxt))
        self.play(LaggedStart(*[GrowArrow(m) for m in msgs], lag_ratio=0.08), run_time=0.72)
        self.play(Write(target))
        self.wait(0.52)
        self.wipe()

    # ------------------------------------------------------------------
    # 10. Outro
    # ------------------------------------------------------------------
    def outro(self):
        a = jtext("Before making the GPU faster", 40, MUTED)
        b = jtext("Redesign the state for the GPU", 55, FG, bold=True)
        c = formula("frontier DP  →  packed  →  rank  →  CUDA", 32, BLUE1)
        group = VGroup(a, b, c).arrange(DOWN, buff=0.30)
        center_in(group, SAFE_RECT, pad=0.55)
        self.audit("outro", group=group)
        self.play(FadeIn(a, shift=UP * 0.08))
        self.play(Write(b), run_time=0.82)
        self.play(Write(c), run_time=1.05)
        self.wait(1.0)
