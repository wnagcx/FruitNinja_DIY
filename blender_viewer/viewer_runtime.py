import argparse
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


MODEL_OBJECT_NAME = "fruit_model"
SLICE_PLANE_NAME = "slice_plane"
VIEWER_COLLECTION_NAME = "FruitNinjaViewer"
POINT_NODE_GROUP_NAME = "FruitPointCloudNodes"
POINT_MATERIAL_NAME = "FruitPointMaterial"
POINT_COLOR_ATTRIBUTE = "viewer_color"
SH_C0 = 0.28209479177387814

WATCH_PATH = ""
POLL_SECONDS = 2.0
BLEND_PATH = ""
AUTO_SAVE_BLEND = False
COLOR_MODE = "raw"
LAST_MTIME = None
LAST_PLANE_STATE = None
SOURCE_COORDS = None
SOURCE_COLORS = None


def parse_runtime_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender runtime for FruitNinja viewer")
    parser.add_argument("--watch-path", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--blend-path", default="")
    parser.add_argument("--auto-save-blend", action="store_true")
    parser.add_argument(
        "--color-mode",
        choices=["raw", "boosted", "grayscale_opacity"],
        default="raw",
    )
    return parser.parse_args(argv)


def get_or_create_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def ensure_object_in_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if obj not in collection.objects[:]:
        collection.objects.link(obj)
    for existing_collection in list(obj.users_collection):
        if existing_collection != collection:
            existing_collection.objects.unlink(obj)


def clear_object(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def clear_mesh(name: str) -> None:
    mesh = bpy.data.meshes.get(name)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def cleanup_previous_model() -> None:
    clear_object(MODEL_OBJECT_NAME)
    clear_mesh(MODEL_OBJECT_NAME)


def parse_vertex_header(filepath: str) -> tuple[int, list[tuple[str, str]], int]:
    vertex_count = None
    properties = []
    in_vertex_element = False
    offset = 0

    with open(filepath, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {filepath}")
            offset += len(line)
            text = line.decode("latin1").strip()

            if text.startswith("format ") and "binary_little_endian" not in text:
                raise RuntimeError(f"Only binary_little_endian PLY is supported: {filepath}")

            if text.startswith("element "):
                _, element_name, count_text = text.split()
                in_vertex_element = element_name == "vertex"
                if in_vertex_element:
                    vertex_count = int(count_text)
                    properties = []
                continue

            if text.startswith("property ") and in_vertex_element:
                _, value_type, property_name = text.split()
                properties.append((property_name, value_type))
                continue

            if text == "end_header":
                break

    if vertex_count is None:
        raise RuntimeError(f"No vertex element found in PLY header: {filepath}")

    return vertex_count, properties, offset


def property_dtype(value_type: str) -> np.dtype:
    mapping = {
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "float64": np.float64,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "char": np.int8,
        "int8": np.int8,
        "ushort": np.uint16,
        "uint16": np.uint16,
        "short": np.int16,
        "int16": np.int16,
        "uint": np.uint32,
        "uint32": np.uint32,
        "int": np.int32,
        "int32": np.int32,
    }
    if value_type not in mapping:
        raise RuntimeError(f"Unsupported PLY property type: {value_type}")
    return mapping[value_type]


def load_ply_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    vertex_count, properties, offset = parse_vertex_header(filepath)
    dtype = np.dtype([(name, property_dtype(value_type)) for name, value_type in properties])

    with open(filepath, "rb") as handle:
        handle.seek(offset)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)

    coords = np.stack(
        [
            data["x"].astype(np.float32, copy=False),
            data["y"].astype(np.float32, copy=False),
            data["z"].astype(np.float32, copy=False),
        ],
        axis=1,
    )

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(data.dtype.names):
        rgb = np.stack(
            [
                data["f_dc_0"].astype(np.float32, copy=False),
                data["f_dc_1"].astype(np.float32, copy=False),
                data["f_dc_2"].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        rgb = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
    else:
        rgb = np.full((vertex_count, 3), 0.8, dtype=np.float32)

    if "opacity" in data.dtype.names:
        opacity = data["opacity"].astype(np.float32, copy=False)
        alpha = (1.0 / (1.0 + np.exp(-opacity))).reshape(-1, 1)
    else:
        alpha = np.ones((vertex_count, 1), dtype=np.float32)

    if COLOR_MODE == "boosted":
        rgb = np.clip((rgb - 0.5) * 1.8 + 0.5, 0.0, 1.0)
    elif COLOR_MODE == "grayscale_opacity":
        rgb = np.repeat(alpha, 3, axis=1)

    colors = np.concatenate([rgb, alpha], axis=1)
    return coords, colors


def center_coords_xy(coords: np.ndarray) -> np.ndarray:
    centered = coords.copy()
    min_xy = centered[:, :2].min(axis=0)
    max_xy = centered[:, :2].max(axis=0)
    centered[:, 0] -= 0.5 * (min_xy[0] + max_xy[0])
    centered[:, 1] -= 0.5 * (min_xy[1] + max_xy[1])
    return centered


def ensure_color_attribute(mesh: bpy.types.Mesh, colors: np.ndarray) -> None:
    color_attribute = mesh.color_attributes.get(POINT_COLOR_ATTRIBUTE)
    if color_attribute is None:
        color_attribute = mesh.color_attributes.new(
            name=POINT_COLOR_ATTRIBUTE,
            type="FLOAT_COLOR",
            domain="POINT",
        )

    if len(colors) == 0:
        return

    color_attribute.data.foreach_set("color", colors.astype(np.float32, copy=False).ravel())


def build_mesh_from_arrays(name: str, coords: np.ndarray, colors: np.ndarray) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    vertex_count = len(coords)
    mesh.vertices.add(vertex_count)
    if vertex_count > 0:
        mesh.vertices.foreach_set("co", coords.astype(np.float32, copy=False).ravel())
    mesh.update()
    ensure_color_attribute(mesh, colors)
    return mesh


def create_point_material() -> bpy.types.Material:
    material = bpy.data.materials.get(POINT_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(name=POINT_MATERIAL_NAME)
        material.use_nodes = True
        node_tree = material.node_tree
        nodes = node_tree.nodes
        links = node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (300, 0)

        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (50, 0)
        principled.inputs["Roughness"].default_value = 0.6

        attribute = nodes.new("ShaderNodeAttribute")
        attribute.location = (-220, 0)
        attribute.attribute_name = POINT_COLOR_ATTRIBUTE

        links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
        links.new(attribute.outputs["Color"], principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = 0.2
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return material


def import_ply(filepath: str, collection: bpy.types.Collection) -> bpy.types.Object:
    cleanup_previous_model()
    coords, colors = load_ply_data(filepath)
    coords = center_coords_xy(coords)
    mesh = build_mesh_from_arrays(MODEL_OBJECT_NAME, coords, colors)
    imported_obj = bpy.data.objects.new(MODEL_OBJECT_NAME, mesh)

    ensure_object_in_collection(imported_obj, collection)
    imported_obj.location = (0.0, 0.0, 0.0)
    imported_obj.rotation_euler = (0.0, 0.0, 0.0)
    imported_obj.scale = (1.0, 1.0, 1.0)
    imported_obj.show_instancer_for_viewport = False
    imported_obj.show_instancer_for_render = False
    return imported_obj


def create_pointcloud_node_group() -> bpy.types.GeometryNodeTree:
    group = bpy.data.node_groups.get(POINT_NODE_GROUP_NAME)
    if group is None:
        group = bpy.data.node_groups.new(POINT_NODE_GROUP_NAME, "GeometryNodeTree")
        group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    group.nodes.clear()
    links = group.links

    node_input = group.nodes.new("NodeGroupInput")
    node_input.location = (-700, 0)

    node_output = group.nodes.new("NodeGroupOutput")
    node_output.location = (-50, 0)

    mesh_to_points = group.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-450, 0)
    mesh_to_points.mode = "VERTICES"
    mesh_to_points.inputs["Radius"].default_value = 0.008

    set_material = group.nodes.new("GeometryNodeSetMaterial")
    set_material.location = (-200, 0)
    set_material.inputs["Material"].default_value = create_point_material()

    links.new(node_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], node_output.inputs["Geometry"])

    return group


def ensure_vertex_instances(model: bpy.types.Object, collection: bpy.types.Collection) -> None:
    node_group = create_pointcloud_node_group()
    modifier = model.modifiers.get("FruitPointCloud")
    if modifier is None:
        modifier = model.modifiers.new(name="FruitPointCloud", type="NODES")
    modifier.node_group = node_group
    model.display_type = "TEXTURED"
    model.hide_render = False


def create_slice_plane(collection: bpy.types.Collection) -> bpy.types.Object:
    plane = bpy.data.objects.get(SLICE_PLANE_NAME)
    if plane is None:
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.active_object
        plane.name = SLICE_PLANE_NAME
        material = bpy.data.materials.get("SlicePlaneMaterial")
        if material is None:
            material = bpy.data.materials.new(name="SlicePlaneMaterial")
            material.use_nodes = True
            principled = material.node_tree.nodes.get("Principled BSDF")
            if principled is not None:
                principled.inputs["Base Color"].default_value = (0.1, 0.6, 1.0, 1.0)
                principled.inputs["Alpha"].default_value = 0.25
            material.blend_method = "BLEND"
        if not plane.data.materials:
            plane.data.materials.append(material)

    ensure_object_in_collection(plane, collection)
    plane.display_type = "TEXTURED"
    return plane


def remove_legacy_slice_volume() -> None:
    legacy = bpy.data.objects.get("slice_volume")
    if legacy is not None:
        bpy.data.objects.remove(legacy, do_unlink=True)


def frame_model(model: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
            if region is None or space is None:
                continue
            with bpy.context.temp_override(area=area, region=region, space_data=space):
                bpy.ops.view3d.view_selected()
            break
    model.select_set(False)


def plane_state(plane: bpy.types.Object) -> tuple[float, ...]:
    location = plane.location
    rotation = plane.rotation_euler
    scale = plane.scale
    return (
        round(location.x, 6),
        round(location.y, 6),
        round(location.z, 6),
        round(rotation.x, 6),
        round(rotation.y, 6),
        round(rotation.z, 6),
        round(scale.x, 6),
        round(scale.y, 6),
        round(scale.z, 6),
    )


def apply_plane_cut(model: bpy.types.Object, plane: bpy.types.Object) -> None:
    global SOURCE_COORDS, SOURCE_COLORS
    if SOURCE_COORDS is None or SOURCE_COLORS is None:
        return

    normal = plane.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    normal.normalize()
    point_on_plane = plane.matrix_world.translation

    normal_np = np.array([normal.x, normal.y, normal.z], dtype=np.float32)
    point_np = np.array([point_on_plane.x, point_on_plane.y, point_on_plane.z], dtype=np.float32)
    signed_distances = (SOURCE_COORDS - point_np) @ normal_np
    mask = signed_distances >= 0.0
    filtered_coords = SOURCE_COORDS[mask]
    filtered_colors = SOURCE_COLORS[mask]

    mesh = model.data
    mesh.clear_geometry()
    kept_vertices = len(filtered_coords)
    if kept_vertices > 0:
        mesh.vertices.add(kept_vertices)
        mesh.vertices.foreach_set("co", filtered_coords.astype(np.float32, copy=False).ravel())
    mesh.update()
    ensure_color_attribute(mesh, filtered_colors)


def setup_scene(model: bpy.types.Object, collection: bpy.types.Collection) -> None:
    global LAST_PLANE_STATE
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    ensure_vertex_instances(model, collection)

    plane = create_slice_plane(collection)
    remove_legacy_slice_volume()
    LAST_PLANE_STATE = plane_state(plane)
    apply_plane_cut(model, plane)

    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(3.0, -3.0, 2.0), rotation=(1.1, 0.0, 0.78))
        scene.camera = bpy.context.active_object

    if "Light" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN", location=(3.0, -3.0, 5.0))

    frame_model(model)


def maybe_save_blend() -> None:
    if AUTO_SAVE_BLEND and BLEND_PATH:
        target = Path(BLEND_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(target))


def describe_source(filepath: str) -> str:
    name = Path(filepath).name
    if name == "gs_fill.ply":
        return "gs_fill"
    if name.startswith("orange_demo_epoch_") and name.endswith(".ply"):
        epoch = name.removeprefix("orange_demo_epoch_").removesuffix(".ply")
        return f"epoch {epoch}"
    return name


def load_model(filepath: str) -> None:
    global SOURCE_COORDS, SOURCE_COLORS
    collection = get_or_create_collection(VIEWER_COLLECTION_NAME)
    model = import_ply(filepath, collection)
    mesh = model.data
    SOURCE_COORDS = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", SOURCE_COORDS.ravel())
    color_attribute = mesh.color_attributes.get(POINT_COLOR_ATTRIBUTE)
    if color_attribute is not None:
        SOURCE_COLORS = np.empty((len(color_attribute.data), 4), dtype=np.float32)
        color_attribute.data.foreach_get("color", SOURCE_COLORS.ravel())
    else:
        SOURCE_COLORS = np.ones((len(mesh.vertices), 4), dtype=np.float32)
    setup_scene(model, collection)
    maybe_save_blend()
    print(f"[FruitNinjaViewer] Loaded model: {filepath}")
    print(f"[FruitNinjaViewer] Active source: {describe_source(filepath)}")


def watch_for_updates() -> float:
    global LAST_MTIME, LAST_PLANE_STATE
    try:
        current_mtime = os.path.getmtime(WATCH_PATH)
    except OSError:
        print(f"[FruitNinjaViewer] Waiting for file: {WATCH_PATH}")
        return POLL_SECONDS

    if LAST_MTIME is None or current_mtime > LAST_MTIME:
        LAST_MTIME = current_mtime
        load_model(WATCH_PATH)
        return POLL_SECONDS

    model = bpy.data.objects.get(MODEL_OBJECT_NAME)
    plane = bpy.data.objects.get(SLICE_PLANE_NAME)
    if model is not None and plane is not None and SOURCE_COORDS is not None:
        current_plane_state = plane_state(plane)
        if LAST_PLANE_STATE != current_plane_state:
            LAST_PLANE_STATE = current_plane_state
            apply_plane_cut(model, plane)

    return POLL_SECONDS


def bootstrap() -> None:
    global WATCH_PATH, POLL_SECONDS, BLEND_PATH, AUTO_SAVE_BLEND, COLOR_MODE
    args = parse_runtime_args()
    WATCH_PATH = args.watch_path
    POLL_SECONDS = max(args.poll_seconds, 0.5)
    BLEND_PATH = args.blend_path
    AUTO_SAVE_BLEND = args.auto_save_blend
    COLOR_MODE = args.color_mode

    print("[FruitNinjaViewer] Runtime started")
    print(f"[FruitNinjaViewer] watch_path={WATCH_PATH}")
    print(f"[FruitNinjaViewer] poll_seconds={POLL_SECONDS}")
    print(f"[FruitNinjaViewer] color_mode={COLOR_MODE}")

    bpy.app.timers.register(watch_for_updates, first_interval=0.1, persistent=True)


bootstrap()
