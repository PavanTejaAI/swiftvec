import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG, PANEL, FG, MUTED, GRID = "#0B1322", "#101B31", "#E2E8F0", "#94A3B8", "#334155"
PALETTE = ["#22D3EE", "#34D399", "#A78BFA", "#F472B6", "#FBBF24"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "grid.alpha": 0.6, "font.size": 12,
})


def style(ax, axis="x"):
    ax.grid(True, axis=axis, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def leaderboard():
    systems = ["Qdrant", "Pinecone", "ChromaDB", "Moss\n(Apple M4 Pro)", "swiftvec\n(Intel i5 laptop)"]
    p50 = [597.6, 432.6, 351.8, 3.1, 2.54]
    colors = ["#475569", "#475569", "#475569", "#64748B", PALETTE[0]]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bars = ax.barh(systems, p50, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_xlabel("end-to-end query P50 in ms, log scale, embedding time included")
    ax.set_title("swiftvec vs published benchmarks: moss protocol, 100k docs, top_k=5",
                 fontsize=15, pad=14, loc="left")
    for b, v in zip(bars, p50):
        label = f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms"
        ax.text(v * 1.12, b.get_y() + b.get_height() / 2, label,
                va="center", color=FG, fontsize=12, fontweight="bold")
    style(ax)
    save(fig, "leaderboard.png")


def load_rows(name):
    rows = []
    path = ROOT / "benchmarks" / "results" / name
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 8 and p[0].isdigit():
            rows.append((int(p[0]), float(p[1]), float(p[2])))
    return rows


def pareto():
    series = [
        ("f32 768d", "f32-768.txt"),
        ("int8 768d + rerank", "int8-768-rerank.txt"),
        ("int8 768d", "int8-768.txt"),
        ("int8 256d MRL + rerank", "int8-256-rerank.txt"),
        ("int8 256d MRL", "int8-256.txt"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, (label, fname) in enumerate(series):
        pts = load_rows(fname)
        c = PALETTE[i]
        ax.plot([p[1] * 100 for p in pts], [p[2] / 1000 for p in pts],
                marker="o", markersize=7, linewidth=2, color=c, label=label)
        for ef, rec, us in pts:
            ax.annotate(f"ef{ef}", (rec * 100, us / 1000), textcoords="offset points",
                        xytext=(6, 6), fontsize=8.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("recall@5 against exact brute-force ground truth, percent")
    ax.set_ylabel("search P50 in ms, log scale")
    ax.set_title("recall vs speed: every point measured against exact ground truth",
                 fontsize=15, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    style(ax, axis="y")
    save(fig, "pareto.png")


def cascade():
    efs = ["ef 64", "ef 128", "ef 256"]
    off = [295, 175, 1167]
    on = [115, 216, 549]
    x = range(len(efs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar([i - w / 2 for i in x], off, w, color="#475569", label="cascade off")
    ax.bar([i + w / 2 for i in x], on, w, color=PALETTE[1], label="cascade on")
    ax.set_xticks(list(x))
    ax.set_xticklabels(efs)
    ax.set_ylabel("search P99 in microseconds")
    ax.set_title("binary cascade: recall unchanged at 1.000, tail latency cut up to 2x",
                 fontsize=14, pad=14, loc="left")
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(i - w / 2, a + 18, str(a), ha="center", color=MUTED, fontsize=10)
        ax.text(i + w / 2, b + 18, str(b), ha="center", color=FG, fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    style(ax, axis="y")
    save(fig, "cascade.png")


def memory():
    tiers = ["f32\n768d", "int8\n768d", "int8 256d\nMRL", "int8 256d\n+ rerank"]
    mb = [307.2, 76.8, 25.6, 128.0]
    colors = ["#475569", "#64748B", PALETTE[1], PALETTE[0]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(tiers, mb, color=colors, width=0.58)
    ax.set_ylabel("vector storage at 100k docs, MB")
    ax.set_title("12x memory reduction from quantization plus Matryoshka truncation",
                 fontsize=14, pad=14, loc="left")
    for b, v in zip(bars, mb):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f} MB",
                ha="center", color=FG, fontsize=11, fontweight="bold")
    style(ax, axis="y")
    save(fig, "memory.png")


if __name__ == "__main__":
    leaderboard()
    pareto()
    cascade()
    memory()
