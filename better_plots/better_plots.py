import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

__all__ = ["set_defaults", "use_science", "fig", "save"]

_DEFAULTS = {
    "linewidth_pt": 426,
    "base_font_at_1": 10,
    "title_scale": 1.15,
    "label_scale": 1.10,
    "tick_scale": 1.00,
    "dpi": 200,
    "use_tex": False,
    "font_family": "serif",
    "output_dir": "paper_figures",
}

def set_defaults(**kwargs):
    for k, v in kwargs.items():
        if k in _DEFAULTS:
            _DEFAULTS[k] = v

def _apply_style(base_font):
    mpl.rcParams.update({
        "figure.dpi": _DEFAULTS["dpi"],
        "savefig.dpi": _DEFAULTS["dpi"],
        "font.size": base_font,
        "axes.titlesize": base_font * _DEFAULTS["title_scale"],
        "axes.labelsize": base_font * _DEFAULTS["label_scale"],
        "xtick.labelsize": base_font * _DEFAULTS["tick_scale"],
        "ytick.labelsize": base_font * _DEFAULTS["tick_scale"],
        "lines.linewidth": 2.0,
        "figure.autolayout": False,
        "text.usetex": _DEFAULTS["use_tex"],
        "font.family": _DEFAULTS["font_family"],
    })

def use_science(*styles):
    try:
        plt.style.use(list(styles) or ["science"])
    except Exception:
        pass

def fig(font: float | None = None, aspect: float = 0.68, legend_out: bool = True):
    plt.style.use("default")

    base_font = _DEFAULTS["base_font_at_1"] if font is None else float(font)
    _apply_style(base_font)

    w_in = _DEFAULTS["linewidth_pt"] / 72.27
    h_in = w_in * aspect
    f, ax = plt.subplots(figsize=(w_in, h_in), layout="constrained")

    if legend_out:
        def place_legend(**kwargs):
            return ax.legend(
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                frameon=False, **kwargs
            )
        ax.place_legend = place_legend
    return f, ax

def save(fig, filename, transparent: bool = False):
    outdir = Path(_DEFAULTS["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, bbox_inches="tight", transparent=transparent)
