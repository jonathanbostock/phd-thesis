# Name space for defaults
from types import SimpleNamespace

defaults = SimpleNamespace(
    fig_width=50 / 25.4,         # 50mm single plot width
    fig_height=40 / 25.4,        # 40mm single plot height
    double_fig_width=2 * 45 / 25.4,  # 2×45mm double plot total width
    double_fig_height=35 / 25.4,     # 35mm double plot height
    small_fig_scale=0.8,
)
