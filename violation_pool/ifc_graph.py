"""IFC → NetworkX graph dönüştürücü.

Her IfcProduct (Project, Site, Building, Storey, Space, Wall, Door, Window,
Slab, Stair, Railing, vb.) bir node olur. Tüm direct attribute'lar ve Pset
property'leri node üzerinde saklanır.

Kenar tipleri:
  aggregates       — IfcRelAggregates (Project→Site→Building→Storey)
  contains         — IfcRelContainedInSpatialStructure (Storey→Wall vs.)
  bounds           — IfcRelSpaceBoundary (Wall→Space)
  voids            — IfcRelVoidsElement (Wall→Opening)
  fills            — IfcRelFillsElement (Door→Opening)
  connects         — IfcRelConnectsPathElements (Wall↔Wall)
  co_bounds_space  — TÜRETİLMİŞ: aynı Space'i sınırlayan iki eleman
                     birbirine bağlanır (via=space_guid).

Saklama: NetworkX node-link JSON. load/save sembolik link gibi çalışır.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


_NODE_TYPES = (
    "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace",
    "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcCovering",
    "IfcDoor", "IfcWindow", "IfcStair", "IfcStairFlight", "IfcRailing",
    "IfcRamp", "IfcColumn", "IfcBeam", "IfcOpeningElement",
)


def _attrs_dict(el) -> dict:
    """Direct IFC attribute'larından çıkarılabilir JSON-uyumlu değerler."""
    out: dict[str, Any] = {}
    for name in ("Name", "Description", "ObjectType", "PredefinedType",
                 "OverallWidth", "OverallHeight", "NominalHeight",
                 "Elevation", "LongName", "Tag"):
        v = getattr(el, name, None)
        if v is None:
            continue
        try:
            json.dumps(v)
            out[name] = v
        except TypeError:
            out[name] = str(v)
    return out


def _psets(el) -> dict:
    try:
        import ifcopenshell.util.element as eu
        return eu.get_psets(el) or {}
    except Exception:
        return {}


def build_graph(ifc_path: str | Path) -> nx.MultiDiGraph:
    import ifcopenshell

    f = ifcopenshell.open(str(ifc_path))
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.graph["ifc_path"] = str(ifc_path)

    # Nodes
    accepted: set[str] = set()
    for t in _NODE_TYPES:
        try:
            elems = f.by_type(t)
        except Exception:
            elems = []
        for el in elems:
            gid = getattr(el, "GlobalId", None)
            if not gid or gid in accepted:
                continue
            g.add_node(
                gid,
                ifc_type=el.is_a(),
                attributes=_attrs_dict(el),
                psets=_psets(el),
            )
            accepted.add(gid)

    def _edge(u, v, **kw):
        if u in g.nodes and v in g.nodes:
            g.add_edge(u, v, **kw)

    # Aggregation
    for rel in f.by_type("IfcRelAggregates"):
        parent = getattr(rel, "RelatingObject", None)
        if not parent:
            continue
        for child in getattr(rel, "RelatedObjects", []) or []:
            _edge(parent.GlobalId, child.GlobalId, rel="aggregates")

    # Containment in spatial structure
    for rel in f.by_type("IfcRelContainedInSpatialStructure"):
        parent = getattr(rel, "RelatingStructure", None)
        if not parent:
            continue
        for child in getattr(rel, "RelatedElements", []) or []:
            _edge(parent.GlobalId, child.GlobalId, rel="contains")

    # Space boundaries — wall bounds space
    space_to_elems: dict[str, list[str]] = {}
    for rel in f.by_type("IfcRelSpaceBoundary"):
        sp = getattr(rel, "RelatingSpace", None)
        el = getattr(rel, "RelatedBuildingElement", None)
        if sp and el:
            _edge(el.GlobalId, sp.GlobalId, rel="bounds")
            space_to_elems.setdefault(sp.GlobalId, []).append(el.GlobalId)

    # Voids / Fills
    for rel in f.by_type("IfcRelVoidsElement"):
        host = getattr(rel, "RelatingBuildingElement", None)
        op = getattr(rel, "RelatedOpeningElement", None)
        if host and op:
            _edge(host.GlobalId, op.GlobalId, rel="voids")
    for rel in f.by_type("IfcRelFillsElement"):
        host = getattr(rel, "RelatedBuildingElement", None)
        op = getattr(rel, "RelatingOpeningElement", None)
        if host and op:
            _edge(host.GlobalId, op.GlobalId, rel="fills")

    # Path element connections
    for rel in f.by_type("IfcRelConnectsPathElements"):
        a = getattr(rel, "RelatingElement", None)
        b = getattr(rel, "RelatedElement", None)
        if a and b:
            _edge(a.GlobalId, b.GlobalId, rel="connects")

    # Derived: co-bounds-space (aynı Space'i sınırlayan elemanlar)
    for sp_id, elems in space_to_elems.items():
        elems = list(dict.fromkeys(elems))  # uniq, sırayı koru
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                _edge(elems[i], elems[j], rel="co_bounds_space", via=sp_id)
                _edge(elems[j], elems[i], rel="co_bounds_space", via=sp_id)

    g.graph["n_nodes"] = g.number_of_nodes()
    g.graph["n_edges"] = g.number_of_edges()
    _embed_layout(g)
    return g


def _embed_layout(g, seed: int = 7) -> None:
    """Spring layout pozisyonlarını node attribute'u olarak göm (x,y).

    Böylece her görselleştirmede yeniden hesaplanmaz, JSON'a kaydedilince
    sonraki açılışlarda layout tutarlı kalır.
    """
    if g.number_of_nodes() == 0:
        return
    simple = nx.Graph()
    simple.add_nodes_from(g.nodes())
    for u, v, _ in g.edges(data=True):
        simple.add_edge(u, v)
    k = 1.4 / max(1, simple.number_of_nodes() ** 0.5)
    pos = nx.spring_layout(simple, seed=seed, k=k)
    for n, (x, y) in pos.items():
        g.nodes[n]["x"] = float(x)
        g.nodes[n]["y"] = float(y)


def save_graph(g: nx.MultiDiGraph, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="links")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="links", multigraph=True, directed=True)


def build_and_save(ifc_path: str | Path, out_path: str | Path) -> Path:
    g = build_graph(ifc_path)
    return save_graph(g, out_path)
