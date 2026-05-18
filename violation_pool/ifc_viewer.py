"""IFC için 3D görselleştirme (plotly mesh).

ifcopenshell.geom ile her IfcProduct tessellate edilir; mesh listesi
plotly.graph_objects.Mesh3d olarak çizilir. Verilen GUID seti varsa
o elemanlar kırmızı + opak vurgulanır (ihlalli elemanlar).

Geometry kernel (OpenCascade) yoksa anlamlı hata mesajı döner; bu durumda
conda-forge kanalından `conda install -c conda-forge ifcopenshell` öneririz.
"""
from __future__ import annotations

from pathlib import Path


def _import_geom():
    import ifcopenshell  # noqa
    import ifcopenshell.geom  # noqa
    return ifcopenshell


def ifc_to_figure(
    ifc_path: str | Path,
    highlight_guids: set[str] | None = None,
    decoy_guids: set[str] | None = None,
    max_elements: int = 5000,
):
    """IFC dosyasından plotly Figure üret. ImportError/RuntimeError'da
    ValueError fırlatır (UI'da gösterilebilsin diye)."""
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ValueError(f"plotly yüklü değil: {e}")
    try:
        ifcopenshell = _import_geom()
        import ifcopenshell.geom as geom
    except Exception as e:
        raise ValueError(
            "ifcopenshell.geom (OpenCascade) kullanılamıyor. "
            "Conda'da: `conda install -c conda-forge ifcopenshell`. "
            f"Detay: {e}"
        )

    try:
        f = ifcopenshell.open(str(ifc_path))
    except Exception as e:
        raise ValueError(f"IFC açılamadı: {e}")
    s = geom.settings()
    try:
        s.set(s.USE_WORLD_COORDS, True)
    except Exception:
        pass

    highlight_guids = set(highlight_guids or [])
    decoy_guids = set(decoy_guids or [])
    meshes = []
    count = 0
    skipped = 0
    for p in f.by_type("IfcProduct"):
        if count >= max_elements:
            break
        if not getattr(p, "Representation", None):
            continue
        try:
            shape = geom.create_shape(s, p)
        except Exception:
            skipped += 1
            continue
        verts = list(shape.geometry.verts)
        faces = list(shape.geometry.faces)
        if not verts or not faces:
            continue
        xs = verts[0::3]
        ys = verts[1::3]
        zs = verts[2::3]
        ii = faces[0::3]
        jj = faces[1::3]
        kk = faces[2::3]
        is_hl = p.GlobalId in highlight_guids
        is_decoy = p.GlobalId in decoy_guids and not is_hl
        if is_hl:
            color, opacity, tag = "#e6194b", 1.0, "[İHLAL]"   # parlak kırmızı
        elif is_decoy:
            color, opacity, tag = "#ffbb33", 1.0, "[DECOY]"   # sarı-turuncu
        else:
            color, opacity, tag = "#9fb3c8", 0.25, ""          # soluk gri
        meshes.append(
            go.Mesh3d(
                x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                color=color, opacity=opacity, flatshading=True,
                name=f"{p.is_a()}",
                hovertext=(
                    f"{tag} {p.is_a()} · {p.GlobalId} · "
                    f"{getattr(p, 'Name', '') or ''}"
                ),
                hoverinfo="text",
                showscale=False,
            )
        )
        count += 1

    if not meshes:
        raise ValueError(
            "Çizilebilecek geometri bulunamadı. "
            "IFC'de IfcExtrudedAreaSolid / IfcShapeRepresentation içeren eleman yok "
            "(LLM 'raw' modunda sık görülür). 'Gerçek IFC içe aktar' veya "
            "parametrik baseline kullan."
        )

    fig = go.Figure(data=meshes)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=620,
    )
    return fig, {"drawn": count, "skipped": skipped}
