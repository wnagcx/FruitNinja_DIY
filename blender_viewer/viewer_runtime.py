import argparse
import os
import sys
from pathlib import Path

import bpy


MODEL_OBJECT_NAME = "fruit_model"
SLICE_PLANE_NAME = "slice_plane"
SLICE_VOLUME_NAME = "slice_volume"
VIEWER_COLLECTION_NAME = "FruitNinjaViewer"

WATCH_PATH = ""
POLL_SECONDS = 2.0
BLEND_PATH = ""
AUTO_SAVE_BLEND = False
LAST_MTIME = None


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
    return imported_obj


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


def create_slice_volume(collection: bpy.types.Collection, plane: bpy.types.Object) -> bpy.types.Object:
    volume = bpy.data.objects.get(SLICE_VOLUME_NAME)
    if volume is None:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
        volume = bpy.context.active_object
        volume.name = SLICE_VOLUME_NAME

    ensure_object_in_collection(volume, collection)
    volume.parent = plane
    volume.matrix_parent_inverse = plane.matrix_world.inverted()
    volume.location = (0.0, 0.0, 50.0)
    volume.rotation_euler = (0.0, 0.0, 0.0)
    volume.scale = (100.0, 100.0, 50.0)
    volume.display_type = "WIRE"
    volume.hide_render = True
    volume.hide_set(False)
    return volume


def ensure_boolean_modifier(model: bpy.types.Object, slice_volume: bpy.types.Object) -> None:
    modifier = model.modifiers.get("FruitNinjaSlice")
    if modifier is None:
        modifier = model.modifiers.new(name="FruitNinjaSlice", type="BOOLEAN")
    modifier.operation = "INTERSECT"
    modifier.solver = "EXACT"
    modifier.object = slice_volume


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


def setup_scene(model: bpy.types.Object, collection: bpy.types.Collection) -> None:
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    plane = create_slice_plane(collection)
    volume = create_slice_volume(collection, plane)
    ensure_boolean_modifier(model, volume)

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


def load_model(filepath: str) -> None:
    collection = get_or_create_collection(VIEWER_COLLECTION_NAME)
    model = import_ply(filepath, collection)
    setup_scene(model, collection)
    maybe_save_blend()
    print(f"[FruitNinjaViewer] Loaded model: {filepath}")


def watch_for_updates() -> float:
    global LAST_MTIME
    try:
        current_mtime = os.path.getmtime(WATCH_PATH)
    except OSError:
        print(f"[FruitNinjaViewer] Waiting for file: {WATCH_PATH}")
        return POLL_SECONDS

    if LAST_MTIME is None or current_mtime > LAST_MTIME:
        LAST_MTIME = current_mtime
        load_model(WATCH_PATH)

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
