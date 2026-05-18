"""Parametrik IFC4 inşaatçısı.

LLM yalnızca küçük bir JSON spec verir (oda boyutları, kapı/pencere konumları).
Bu modül ifcopenshell ile geometriye sahip geçerli bir IFC4 dosyası inşa
eder; böylece ifcopenshell.geom her elemanı tessellate edebilir ve graph
zengin olur.

Spec şeması:
{
  "name": "Villa-1",
  "storey_height": 3.0,
  "wall_thickness": 0.20,
  "rooms": [{"name": "Salon", "origin": [0,0], "size": [5,4]}, ...],
  "openings": [
    {"room": "Salon", "side": "south|north|east|west",
     "type": "door|window", "width": 1.0, "height": 2.2,
     "sill": 0.0, "offset": 1.2}
  ]
}
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import ifcopenshell
import ifcopenshell.guid as ifc_guid


# ---------------- helpers ----------------
def _dir(f, *components):
    return f.create_entity("IfcDirection",
                           DirectionRatios=tuple(float(c) for c in components))


def _pt(f, x, y, z=0.0):
    return f.create_entity("IfcCartesianPoint",
                           Coordinates=(float(x), float(y), float(z)))


def _axis(f, origin, z_axis=(0, 0, 1), x_axis=(1, 0, 0)):
    return f.create_entity(
        "IfcAxis2Placement3D",
        Location=origin,
        Axis=_dir(f, *z_axis),
        RefDirection=_dir(f, *x_axis),
    )


def _local_placement(f, origin, z_axis=(0, 0, 1), x_axis=(1, 0, 0), rel_to=None):
    return f.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=rel_to,
        RelativePlacement=_axis(f, origin, z_axis, x_axis),
    )


def _rect_extrude_solid(f, length, thickness, height, direction=(0, 0, 1)):
    """Centered rectangle profile (length × thickness) extruded by `height`
    along `direction` in the local frame.
    """
    profile = f.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA", XDim=float(length), YDim=float(thickness),
    )
    axis = _axis(f, _pt(f, 0, 0, 0), (0, 0, 1), (1, 0, 0))
    return f.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=axis,
        ExtrudedDirection=_dir(f, *direction),
        Depth=float(height),
    )


def _product_shape(f, body_ctx, solid):
    rep = f.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_ctx,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )
    return f.create_entity("IfcProductDefinitionShape", Representations=[rep])


def _setup_project(f, name):
    person = f.create_entity("IfcPerson", FamilyName="Generator")
    org = f.create_entity("IfcOrganization", Name="ViolationPool")
    po = f.create_entity("IfcPersonAndOrganization",
                         ThePerson=person, TheOrganization=org)
    app = f.create_entity(
        "IfcApplication",
        ApplicationDeveloper=org, Version="1.0",
        ApplicationFullName="ViolationPool", ApplicationIdentifier="vp",
    )
    owner = f.create_entity(
        "IfcOwnerHistory",
        OwningUser=po, OwningApplication=app,
        State="READWRITE", ChangeAction="ADDED",
        CreationDate=int(time.time()),
    )

    world_axis = _axis(f, _pt(f, 0, 0, 0))
    true_north = f.create_entity("IfcDirection", DirectionRatios=(0.0, 1.0))
    world_ctx = f.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        Precision=1e-5,
        WorldCoordinateSystem=world_axis,
        TrueNorth=true_north,
    )
    body_ctx = f.create_entity(
        "IfcGeometricRepresentationSubContext",
        ContextIdentifier="Body", ContextType="Model",
        ParentContext=world_ctx, TargetView="MODEL_VIEW",
    )

    length_u = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    angle_u = f.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
    area_u = f.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
    volume_u = f.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    units = f.create_entity(
        "IfcUnitAssignment",
        Units=[length_u, angle_u, area_u, volume_u],
    )

    project = f.create_entity(
        "IfcProject",
        GlobalId=ifc_guid.new(), OwnerHistory=owner, Name=name,
        RepresentationContexts=[world_ctx], UnitsInContext=units,
    )
    return project, owner, body_ctx


def _make_spatial(f, owner, cls, name, parent_placement=None):
    placement = _local_placement(f, _pt(f, 0, 0, 0), rel_to=parent_placement)
    kwargs = dict(
        GlobalId=ifc_guid.new(), OwnerHistory=owner, Name=name,
        ObjectPlacement=placement,
    )
    if cls in ("IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"):
        kwargs["CompositionType"] = "ELEMENT"
    if cls == "IfcBuildingStorey":
        kwargs["Elevation"] = 0.0
    return f.create_entity(cls, **kwargs), placement


def _aggregate(f, owner, parent, children):
    if not children:
        return
    f.create_entity(
        "IfcRelAggregates",
        GlobalId=ifc_guid.new(), OwnerHistory=owner,
        RelatingObject=parent, RelatedObjects=list(children),
    )


def _contain(f, owner, structure, products):
    if not products:
        return
    f.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifc_guid.new(), OwnerHistory=owner,
        RelatingStructure=structure, RelatedElements=list(products),
    )


def _bound(f, owner, space, element):
    f.create_entity(
        "IfcRelSpaceBoundary",
        GlobalId=ifc_guid.new(), OwnerHistory=owner,
        RelatingSpace=space, RelatedBuildingElement=element,
        PhysicalOrVirtualBoundary="PHYSICAL",
        InternalOrExternalBoundary="INTERNAL",
    )


def _wall_between(f, owner, body_ctx, name, start, end, height, thickness,
                  rel_to):
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy)
    if length <= 0:
        return None
    mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
    dx, dy = (ex - sx) / length, (ey - sy) / length
    placement = _local_placement(
        f, _pt(f, mx, my, 0),
        z_axis=(0, 0, 1), x_axis=(dx, dy, 0),
        rel_to=rel_to,
    )
    solid = _rect_extrude_solid(f, length, thickness, height)
    shape = _product_shape(f, body_ctx, solid)
    return f.create_entity(
        "IfcWallStandardCase",
        GlobalId=ifc_guid.new(), OwnerHistory=owner, Name=name,
        ObjectPlacement=placement, Representation=shape,
    )


def _slab(f, owner, body_ctx, name, footprint, thickness, rel_to):
    xmin, ymin, xmax, ymax = footprint
    length = xmax - xmin
    width = ymax - ymin
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    placement = _local_placement(f, _pt(f, cx, cy, 0), rel_to=rel_to)
    solid = _rect_extrude_solid(f, length, width, thickness,
                                direction=(0, 0, -1))
    shape = _product_shape(f, body_ctx, solid)
    return f.create_entity(
        "IfcSlab",
        GlobalId=ifc_guid.new(), OwnerHistory=owner, Name=name,
        ObjectPlacement=placement, Representation=shape,
        PredefinedType="FLOOR",
    )


def _make_opening_element(f, owner, body_ctx, cls, name,
                          world_xy, base_z, dx, dy,
                          width, height, panel_t, rel_to):
    placement = _local_placement(
        f, _pt(f, world_xy[0], world_xy[1], base_z),
        z_axis=(0, 0, 1), x_axis=(dx, dy, 0),
        rel_to=rel_to,
    )
    solid = _rect_extrude_solid(f, width, panel_t, height)
    shape = _product_shape(f, body_ctx, solid)
    kwargs = dict(
        GlobalId=ifc_guid.new(), OwnerHistory=owner, Name=name,
        ObjectPlacement=placement, Representation=shape,
        OverallWidth=float(width), OverallHeight=float(height),
    )
    return f.create_entity(cls, **kwargs)


def build_house_ifc(spec: dict, out_path: str | Path) -> Path:
    """Spec'i tüketip valid IFC4 dosyası yazar; out_path döner."""
    f = ifcopenshell.file(schema="IFC4")
    project, owner, body_ctx = _setup_project(f, spec.get("name", "House"))

    site, site_pl = _make_spatial(f, owner, "IfcSite", "Site")
    building, bld_pl = _make_spatial(f, owner, "IfcBuilding", "Building", site_pl)
    storey, st_pl = _make_spatial(f, owner, "IfcBuildingStorey", "Storey 1", bld_pl)
    _aggregate(f, owner, project, [site])
    _aggregate(f, owner, site, [building])
    _aggregate(f, owner, building, [storey])

    rooms = spec.get("rooms") or []
    h = float(spec.get("storey_height", 3.0))
    t = float(spec.get("wall_thickness", 0.20))

    physical: list = []
    spaces: list = []
    walls_by_room: dict[str, list] = {}

    if rooms:
        xs = [r["origin"][0] for r in rooms] + [r["origin"][0] + r["size"][0] for r in rooms]
        ys = [r["origin"][1] for r in rooms] + [r["origin"][1] + r["size"][1] for r in rooms]
        slab = _slab(
            f, owner, body_ctx, "Floor",
            (min(xs) - 0.3, min(ys) - 0.3, max(xs) + 0.3, max(ys) + 0.3),
            0.20, st_pl,
        )
        physical.append(slab)

    for room in rooms:
        rname = room["name"]
        space, _ = _make_spatial(f, owner, "IfcSpace", rname, st_pl)
        spaces.append(space)
        walls_by_room[rname] = []
        x0, y0 = room["origin"]
        w, l = room["size"]
        edges = [
            ("south", (x0, y0),         (x0 + w, y0)),
            ("east",  (x0 + w, y0),     (x0 + w, y0 + l)),
            ("north", (x0 + w, y0 + l), (x0, y0 + l)),
            ("west",  (x0, y0 + l),     (x0, y0)),
        ]
        for side, p1, p2 in edges:
            wall = _wall_between(f, owner, body_ctx, f"{rname}-{side}",
                                  p1, p2, h, t, st_pl)
            if wall is None:
                continue
            walls_by_room[rname].append((side, wall, p1, p2))
            physical.append(wall)
            _bound(f, owner, space, wall)

    _aggregate(f, owner, storey, spaces)

    # Doors / Windows
    extras = []
    for op in (spec.get("openings") or []):
        rname = op.get("room")
        side = op.get("side")
        if rname not in walls_by_room:
            continue
        match = [item for item in walls_by_room[rname] if item[0] == side]
        if not match:
            continue
        _, _wall, (sx, sy), (ex, ey) = match[0]
        wlen = math.hypot(ex - sx, ey - sy)
        if wlen <= 0:
            continue
        width = float(op.get("width", 1.0 if op.get("type") == "door" else 1.5))
        height = float(op.get("height", 2.2 if op.get("type") == "door" else 1.5))
        sill = float(op.get("sill", 0.0 if op.get("type") == "door" else 0.9))
        offset = float(op.get("offset", (wlen - width) / 2.0))
        offset = max(0.05, min(wlen - width - 0.05, offset))
        dx, dy = (ex - sx) / wlen, (ey - sy) / wlen
        cx = sx + dx * (offset + width / 2.0)
        cy = sy + dy * (offset + width / 2.0)
        cls = "IfcWindow" if op.get("type") == "window" else "IfcDoor"
        elem = _make_opening_element(
            f, owner, body_ctx, cls,
            op.get("name", f"{cls[3:]}-{rname}-{side}"),
            (cx, cy), sill, dx, dy,
            width, height, panel_t=0.05, rel_to=st_pl,
        )
        extras.append(elem)

    _contain(f, owner, storey, physical + extras)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out_p))
    return out_p


# Varsayılan örnek spec (LLM cevap veremezse fallback)
EXAMPLE_SPEC = {
    "name": "Villa-Sample",
    "storey_height": 3.0,
    "wall_thickness": 0.20,
    "rooms": [
        {"name": "Salon",       "origin": [0.0, 0.0], "size": [5.0, 4.0]},
        {"name": "Mutfak",      "origin": [5.0, 0.0], "size": [3.5, 4.0]},
        {"name": "Yatak Odasi", "origin": [0.0, 4.0], "size": [4.0, 4.0]},
        {"name": "Banyo",       "origin": [4.0, 4.0], "size": [3.0, 4.0]},
    ],
    "openings": [
        {"room": "Salon",       "side": "south", "type": "door",
         "width": 1.10, "height": 2.20, "offset": 1.50},
        {"room": "Mutfak",      "side": "east",  "type": "window",
         "width": 1.50, "height": 1.50, "sill": 0.90, "offset": 1.00},
        {"room": "Yatak Odasi", "side": "north", "type": "window",
         "width": 1.50, "height": 1.50, "sill": 0.90, "offset": 1.00},
        {"room": "Banyo",       "side": "east",  "type": "window",
         "width": 1.30, "height": 1.20, "sill": 1.20, "offset": 0.80},
    ],
}
