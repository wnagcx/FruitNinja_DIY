import argparse
import os
import sys
from array import array
from pathlib import Path

import bpy
from mathutils import Vector


MODEL_OBJECT_NAME = "fruit_model"
SLICE_PLANE_NAME = "slice_plane"
VIEWER_COLLECTION_NAME = "FruitNinjaViewer"
POINT_NODE_GROUP_NAME = "FruitPointCloudNodes"

WATCH_PATH = ""
POLL_SECONDS = 2.0
BLEND_PATH = ""
AUTO_SAVE_BLEND = False
LAST_MTIME = None
LAST_PLANE_STATE = None
SOURCE_COORDS = None


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


def import_ply(filepath: str, collection: bpy.types.Collection) -> bpy.types.Object:
    cleanup_previous_model()
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=filepath)
    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    if not imported_names:
        raise RuntimeError(f"PLY import created no object: {filepath}")

    imported_obj = bpy.data.objects[imported_names[0]]
    imported_obj.name = MODEL_OBJECT_NAME
    if imported_obj.data is not None:
        imported_obj.data.name = MODEL_OBJECT_NAME

    ensure_object_in_collection(imported_obj, collection)
    imported_obj.location = (0.0, 0.0, 0.0)
    imported_obj.rotation_euler = (0.0, 0.0, 0.0)
    imported_obj.scale = (1.0, 1.0, 1.0)
    imported_obj.show_instancer_for_viewport = False
    imported_obj.show_instancer_for_render = False
    return imported_obj


def center_model_xy(model: bpy.types.Object) -> None:
    if model.data is None or len(model.data.vertices) == 0:
        return

    bbox_min_x = min(v.co.x for v in model.data.vertices)
    bbox_max_x = max(v.co.x for v in model.data.vertices)
    bbox_min_y = min(v.co.y for v in model.data.vertices)
    bbox_max_y = max(v.co.y for v in model.data.vertices)

    offset_x = -0.5 * (bbox_min_x + bbox_max_x)
    offset_y = -0.5 * (bbox_min_y + bbox_max_y)

    if abs(offset_x) < 1e-9 and abs(offset_y) < 1e-9:
        return

    for vert in model.data.vertices:
        vert.co.x += offset_x
        vert.co.y += offset_y
    model.data.update()


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

    links.new(node_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], node_output.inputs["Geometry"])

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


def extract_source_coords(model: bpy.types.Object) -> array:
    vertex_count = len(model.data.vertices)
    coords = array("f", [0.0]) * (vertex_count * 3)
    model.data.vertices.foreach_get("co", coords)
    return coords


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
    global SOURCE_COORDS
    if SOURCE_COORDS is None:
        return

    normal = plane.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    normal.normalize()
    point_on_plane = plane.matrix_world.translation

    filtered = array("f")
    coords = SOURCE_COORDS
    for i in range(0, len(coords), 3):
        x = coords[i]
        y = coords[i + 1]
        z = coords[i + 2]
        signed_distance = (
            normal.x * (x - point_on_plane.x)
            + normal.y * (y - point_on_plane.y)
            + normal.z * (z - point_on_plane.z)
        )
        if signed_distance >= 0.0:
            filtered.extend((x, y, z))

    mesh = model.data
    mesh.clear_geometry()
    kept_vertices = len(filtered) // 3
    if kept_vertices > 0:
        mesh.vertices.add(kept_vertices)
        mesh.vertices.foreach_set("co", filtered)
    mesh.update()


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
    global SOURCE_COORDS
    collection = get_or_create_collection(VIEWER_COLLECTION_NAME)
    model = import_ply(filepath, collection)
    center_model_xy(model)
    SOURCE_COORDS = extract_source_coords(model)
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
    global WATCH_PATH, POLL_SECONDS, BLEND_PATH, AUTO_SAVE_BLEND
    args = parse_runtime_args()
    WATCH_PATH = args.watch_path
    POLL_SECONDS = max(args.poll_seconds, 0.5)
    BLEND_PATH = args.blend_path
    AUTO_SAVE_BLEND = args.auto_save_blend

    print("[FruitNinjaViewer] Runtime started")
    print(f"[FruitNinjaViewer] watch_path={WATCH_PATH}")
    print(f"[FruitNinjaViewer] poll_seconds={POLL_SECONDS}")

    bpy.app.timers.register(watch_for_updates, first_interval=0.1, persistent=True)


bootstrap()
