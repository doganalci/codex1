"""Plotly tabanlı graph görselleştirici (2D spring layout).

Kenarlar `rel` tipine göre renklenir; highlight_guids set'i verilirse o
node'lar büyük, opak ve kırmızı vurgulanır (ihlal edilen/eklenen elemanlar).
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx


_EDGE_COLORS = {
    "aggregates":      "#7f7f7f",
    "contains":        "#1f77b4",
    "bounds":          "#2ca02c",
    "voids":           "#ff7f0e",
    "fills":           "#9467bd",
    "connects":        "#17becf",
    "co_bounds_space": "#bcbd22",
}


def graph_to_figure(
    g: nx.MultiDiGraph,
    highlight_guids: Iterable[str] | None = None,
    decoy_guids: Iterable[str] | None = None,
    seed: int = 7,
):
    import plotly.graph_objects as go

    hl = set(highlight_guids or [])
    dc = set(decoy_guids or []) - hl  # highlight her zaman önce gelir

    if g.number_of_nodes() == 0:
        raise ValueError("Boş graf.")

    # Her durumda undirected basit kopya üret (degree hesabı için).
    simple = nx.Graph()
    simple.add_nodes_from(g.nodes(data=True))
    for u, v, d in g.edges(data=True):
        simple.add_edge(u, v, rel=d.get("rel"))

    # Pozisyonlar graph'ta gömülü mü? (ifc_graph._embed_layout)
    has_pos = all(
        "x" in g.nodes[n] and "y" in g.nodes[n] for n in g.nodes
    )
    if has_pos:
        pos = {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes}
    else:
        pos = nx.spring_layout(
            simple, seed=seed,
            k=1.4 / max(1, simple.number_of_nodes() ** 0.5),
        )

    # Edges: gruplandırılmış (her rel tipi ayrı trace)
    edge_traces = []
    by_rel: dict[str, list[tuple]] = {}
    for u, v, d in g.edges(data=True):
        by_rel.setdefault(d.get("rel", "?"), []).append((u, v))
    for rel, pairs in by_rel.items():
        xs: list[float] = []
        ys: list[float] = []
        for u, v in pairs:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=1, color=_EDGE_COLORS.get(rel, "#cccccc")),
            hoverinfo="none",
            name=rel,
            opacity=0.65,
        ))

    # Nodes
    node_x, node_y, txt, hover, sizes, colors, borders = [], [], [], [], [], [], []
    deg = dict(simple.degree())
    for n, d in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        is_hl = n in hl
        is_decoy = n in dc
        attrs = d.get("attributes", {}) or {}
        psets = d.get("psets", {}) or {}
        h = f"<b>{d.get('ifc_type', '?')}</b><br>guid={n}"
        if attrs.get("Name"):
            h += f"<br>name={attrs['Name']}"
        if attrs:
            h += "<br>" + "<br>".join(f"{k}={v}" for k, v in attrs.items() if k != "Name")
        if psets:
            ps_short = "; ".join(list(psets.keys())[:5])
            h += f"<br>psets: {ps_short}"
        if is_hl:
            h = "[İHLAL] " + h
        elif is_decoy:
            h = "[DECOY] " + h
        hover.append(h)
        label = d.get("ifc_type", "")
        if label.startswith("Ifc"):
            label = label[3:]
        txt.append(label)
        base = 8 + min(deg.get(n, 0), 20)
        if is_hl:
            sizes.append(base + 14)
            colors.append("#e6194b")  # parlak kırmızı
            borders.append("#000000")
        elif is_decoy:
            sizes.append(base + 10)
            colors.append("#ffbb33")  # sarı-turuncu
            borders.append("#7a4d00")
        else:
            sizes.append(base)
            colors.append("#4c78a8")  # soluk mavi
            borders.append("#1f3a5f")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=txt, textposition="top center",
        textfont=dict(size=9, color="#dddddd"),
        marker=dict(size=sizes, color=colors,
                    line=dict(width=1.3, color=borders)),
        hovertext=hover, hoverinfo="text",
        name="nodes",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=680,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        dragmode="pan",   # mouse drag = pan
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig
