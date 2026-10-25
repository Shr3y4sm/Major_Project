"""Generic neuroanatomy visual for surgical-planning education.

The scene is intentionally illustrative. It is not registered to any patient,
does not represent a patient's vessels or tracts, and must never be used for
navigation or clinical distance measurements.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def generic_neuroanatomy_figure(show_simulated_tumour: bool = True) -> go.Figure:
    """Return a labelled generic teaching scene, separate from patient space.

    ``show_simulated_tumour`` adds a deliberately fictional lesion marker for
    teaching spatial concepts; it has no patient registration or measurement.
    """
    fig = go.Figure()
    u, v = np.mgrid[0:2 * np.pi:45j, 0:np.pi:28j]
    x = 78 * np.cos(u) * np.sin(v)
    y = 58 * np.sin(u) * np.sin(v)
    z = 52 * np.cos(v)
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=.10, showscale=False,
                             colorscale=[[0, "#94a3b8"], [1, "#94a3b8"]],
                             name="Generic brain envelope", hoverinfo="skip"))
    paths = {
        "Corticospinal tract (illustrative)": ("#38bdf8", [(-32, -12, -42), (-18, -8, -20), (-8, -4, 5), (4, 0, 26)]),
        "Arcuate fasciculus (illustrative)": ("#fbbf24", [(-45, 18, 4), (-25, 25, 25), (2, 20, 30), (25, 12, 12)]),
        "Major arterial course (illustrative)": ("#fb7185", [(-52, 0, -5), (-28, -9, -2), (0, 0, 4), (28, 9, 10), (52, 15, 22)]),
    }
    for name, (color, points) in paths.items():
        values = np.asarray(points)
        fig.add_trace(go.Scatter3d(x=values[:, 0], y=values[:, 1], z=values[:, 2],
                                   mode="lines", name=name,
                                   line=dict(color=color, width=8)))
    regions = {"Motor cortex": (-20, -42, 30), "Language network": (-52, 17, 14),
               "Visual pathway": (35, -15, -18)}
    fig.add_trace(go.Scatter3d(x=[p[0] for p in regions.values()],
                               y=[p[1] for p in regions.values()],
                               z=[p[2] for p in regions.values()],
                               mode="markers+text", text=list(regions), textposition="top center",
                               marker=dict(size=6, color="#a78bfa"), name="Illustrative regions"))
    if show_simulated_tumour:
        a, b = np.mgrid[0:2 * np.pi:26j, 0:np.pi:18j]
        tumour_x = -18 + 16 * np.cos(a) * np.sin(b)
        tumour_y = 13 + 13 * np.sin(a) * np.sin(b)
        tumour_z = 8 + 14 * np.cos(b)
        fig.add_trace(go.Surface(
            x=tumour_x, y=tumour_y, z=tumour_z, opacity=.82, showscale=False,
            colorscale=[[0, "#c084fc"], [1, "#7e22ce"]], name="Simulated tumour (teaching only)"))
    fig.update_layout(title="Generic neuroanatomy — simulated teaching model only",
                      paper_bgcolor="#202023", plot_bgcolor="#202023",
                      font=dict(color="#f5f5f7"), margin=dict(l=0, r=0, b=0, t=48),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      scene=dict(bgcolor="#202023", aspectmode="data",
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    return fig
