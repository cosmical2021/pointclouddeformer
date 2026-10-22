bl_info = {
    "name": "Point Cloud Editable Vertex Proxy",
    "author": "Gemini",
    "version": (2, 0, 0),
    "blender": (5, 3, 0),
    "location": "View3D > Sidebar (N) > PC Proxy",
    "description": "Generates an editable vertex proxy from a Point Cloud and deforms the Point Cloud via Geometry Nodes",
    "category": "Rigging",
}

import bpy
import bmesh
from mathutils import Vector, Matrix


# ------------------------------------------------------------------------
# Helper: Extract World-Space Coordinates from PointCloud or Mesh
# ------------------------------------------------------------------------
def get_pointcloud_world_coords(obj, max_samples=2000):
    """Extracts a subsampled list of world-space Vector coordinates from a PointCloud or Mesh."""
    mw = obj.matrix_world
    coords = []

    if obj.type == 'MESH':
        verts = obj.data.vertices
        total = len(verts)
        step = max(1, total // max_samples)
        coords = [mw @ verts[i].co for i in range(0, total, step)]
    elif obj.type == 'POINTCLOUD':
        pts = obj.data.points
        total = len(pts)
        step = max(1, total // max_samples)
        coords = [mw @ pts[i].co for i in range(0, total, step)]

    if not coords:
        # Fallback to object bounding box corners
        coords = [mw @ Vector(corner) for corner in obj.bound_box]

    return coords


# ------------------------------------------------------------------------
# Helper: Generate Editable Proxy Mesh from Point Cloud
# ------------------------------------------------------------------------
def generate_editable_proxy_mesh(pc_obj, proxy_type='HULL', subdivisions=2, padding=0.05, sample_limit=1500):
    """Builds a BMesh proxy around the point cloud points in the Point Cloud's local space."""
    bm = bmesh.new()
    mw_inv = pc_obj.matrix_world.inverted()
    world_pts = get_pointcloud_world_coords(pc_obj, max_samples=sample_limit)
    local_pts = [mw_inv @ p for p in world_pts]

    if proxy_type == 'HULL' and len(local_pts) >= 4:
        for co in local_pts:
            bm.verts.new(co)
        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=bm.verts[:])

        # Dissolve tiny degenerate triangles and remesh evenly via subdivide
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=max(padding, 0.02))

        # Push vertices outward slightly by padding
        bm.normal_update()
        for v in bm.verts:
            v.co += v.normal * padding

        if subdivisions > 0:
            bmesh.ops.subdivide_edges(
                bm,
                edges=bm.edges[:],
                cuts=subdivisions,
                use_grid_fill=True,
            )

    elif proxy_type == 'SUBSAMPLED_VERTS':
        # Pure vertex cloud + wireframe links for direct vertex pulling
        for co in local_pts:
            bm.verts.new(co)
        bm.verts.ensure_lookup_table()
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=max(padding, 0.02))
        # Add convex hull edges so normals exist for rotation interpolation
        if len(bm.verts) >= 4:
            bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)

    else:  # 'BOX' (Subdivided Bounding Cage)
        min_v = Vector((min(p.x for p in local_pts), min(p.y for p in local_pts), min(p.z for p in local_pts)))
        max_v = Vector((max(p.x for p in local_pts), max(p.y for p in local_pts), max(p.z for p in local_pts)))
        min_v -= Vector((padding, padding, padding))
        max_v += Vector((padding, padding, padding))

        center = (min_v + max_v) * 0.5
        size = max_v - min_v
        scale_mat = Matrix.Diagonal((max(size.x * 0.5, 0.01), max(size.y * 0.5, 0.01), max(size.z * 0.5, 0.01), 1.0))
        xform = Matrix.Translation(center) @ scale_mat

        bmesh.ops.create_cube(bm, size=2.0, matrix=xform)
        if subdivisions > 0:
            bmesh.ops.subdivide_edges(
                bm,
                edges=bm.edges[:],
                cuts=subdivisions,
                use_grid_fill=True,
            )

    mesh_data = bpy.data.meshes.new(f"{pc_obj.name}_EditableProxyMesh")
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()
    return mesh_data


# ------------------------------------------------------------------------
# Helper: Build Geometry Nodes Tree for Vertex Proxy Deformation
# ------------------------------------------------------------------------
def build_vertex_proxy_nodetree(node_group_name):
    """
    Constructs a Geometry Nodes modifier that deforms the Point Cloud
    based on vertex & normal deltas between Rest Proxy and Editable Proxy.
    """
    if node_group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[node_group_name])

    ng = bpy.data.node_groups.new(name=node_group_name, type='GeometryNodeTree')
    ng.is_modifier = True

    iface = ng.interface
    iface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    iface.new_socket(name="Rest Proxy", in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(name="Editable Proxy", in_out='INPUT', socket_type='NodeSocketObject')

    in_inf = iface.new_socket(name="Influence", in_out='INPUT', socket_type='NodeSocketFloat')
    in_inf.default_value = 1.0
    in_inf.min_value = 0.0
    in_inf.max_value = 1.0

    in_rot = iface.new_socket(name="Use Normal Rotation", in_out='INPUT', socket_type='NodeSocketFloat')
    in_rot.default_value = 1.0
    in_rot.min_value = 0.0
    in_rot.max_value = 1.0

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new('NodeGroupInput')
    n_in.location = (-950, 0)
    n_out = nodes.new('NodeGroupOutput')
    n_out.location = (950, 0)

    # Object Info: Rest Proxy & Editable Proxy in RELATIVE space
    n_rest_info = nodes.new('GeometryNodeObjectInfo')
    n_rest_info.transform_space = 'RELATIVE'
    n_rest_info.location = (-700, -150)

    n_edit_info = nodes.new('GeometryNodeObjectInfo')
    n_edit_info.transform_space = 'RELATIVE'
    n_edit_info.location = (-700, -400)

    links.new(n_in.outputs["Rest Proxy"], n_rest_info.inputs["Object"])
    links.new(n_in.outputs["Editable Proxy"], n_edit_info.inputs["Object"])

    n_pos = nodes.new('GeometryNodeInputPosition')
    n_pos.location = (-700, 150)

    n_norm = nodes.new('GeometryNodeInputNormal')
    n_norm.location = (-700, 50)

    # Find nearest vertex index on Rest Proxy
    n_nearest = nodes.new('GeometryNodeSampleNearest')
    n_nearest.domain = 'POINT'
    n_nearest.location = (-420, -200)
    links.new(n_rest_info.outputs["Geometry"], n_nearest.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_nearest.inputs["Sample Position"])

    # Sample Rest Vertex Position & Normal
    n_rest_vpos = nodes.new('GeometryNodeSampleIndex')
    n_rest_vpos.data_type = 'FLOAT_VECTOR'
    n_rest_vpos.domain = 'POINT'
    n_rest_vpos.location = (-150, -100)
    links.new(n_rest_info.outputs["Geometry"], n_rest_vpos.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_rest_vpos.inputs["Value"])
    links.new(n_nearest.outputs["Index"], n_rest_vpos.inputs["Index"])

    # Sample Editable Vertex Position & Normal at the SAME Vertex Index
    n_edit_vpos = nodes.new('GeometryNodeSampleIndex')
    n_edit_vpos.data_type = 'FLOAT_VECTOR'
    n_edit_vpos.domain = 'POINT'
    n_edit_vpos.location = (-150, -300)
    links.new(n_edit_info.outputs["Geometry"], n_edit_vpos.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_edit_vpos.inputs["Value"])
    links.new(n_nearest.outputs["Index"], n_edit_vpos.inputs["Index"])

    n_rest_vnorm = nodes.new('GeometryNodeSampleIndex')
    n_rest_vnorm.data_type = 'FLOAT_VECTOR'
    n_rest_vnorm.domain = 'POINT'
    n_rest_vnorm.location = (-150, -500)
    links.new(n_rest_info.outputs["Geometry"], n_rest_vnorm.inputs["Geometry"])
    links.new(n_norm.outputs["Normal"], n_rest_vnorm.inputs["Value"])
    links.new(n_nearest.outputs["Index"], n_rest_vnorm.inputs["Index"])

    n_edit_vnorm = nodes.new('GeometryNodeSampleIndex')
    n_edit_vnorm.data_type = 'FLOAT_VECTOR'
    n_edit_vnorm.domain = 'POINT'
    n_edit_vnorm.location = (-150, -700)
    links.new(n_edit_info.outputs["Geometry"], n_edit_vnorm.inputs["Geometry"])
    links.new(n_norm.outputs["Normal"], n_edit_vnorm.inputs["Value"])
    links.new(n_nearest.outputs["Index"], n_edit_vnorm.inputs["Index"])

    # Offset Vector = Point_Pos - Rest_Vertex_Pos
    n_sub_offset = nodes.new('ShaderNodeVectorMath')
    n_sub_offset.operation = 'SUBTRACT'
    n_sub_offset.location = (120, 50)
    links.new(n_pos.outputs["Position"], n_sub_offset.inputs[0])
    links.new(n_rest_vpos.outputs["Value"], n_sub_offset.inputs[1])

    # Compute Normal Rotation Delta
    n_align_edit = nodes.new('FunctionNodeAlignRotationToVector')
    n_align_edit.location = (120, -450)
    links.new(n_edit_vnorm.outputs["Value"], n_align_edit.inputs["Vector"])

    n_align_rest = nodes.new('FunctionNodeAlignRotationToVector')
    n_align_rest.location = (120, -650)
    links.new(n_rest_vnorm.outputs["Value"], n_align_rest.inputs["Vector"])

    n_inv_rest = nodes.new('FunctionNodeInvertRotation')
    n_inv_rest.location = (300, -650)
    links.new(n_align_rest.outputs["Rotation"], n_inv_rest.inputs["Rotation"])

    n_rot_delta = nodes.new('FunctionNodeRotateRotation')
    n_rot_delta.location = (460, -500)
    links.new(n_inv_rest.outputs["Rotation"], n_rot_delta.inputs["Rotation"])
    links.new(n_align_edit.outputs["Rotation"], n_rot_delta.inputs["Rotate By"])

    # Rotate Offset Vector
    n_rot_vec = nodes.new('FunctionNodeRotateVector')
    n_rot_vec.location = (460, -150)
    links.new(n_sub_offset.outputs["Vector"], n_rot_vec.inputs["Vector"])
    links.new(n_rot_delta.outputs["Rotation"], n_rot_vec.inputs["Rotation"])

    # Blend between unrotated offset and rotated offset
    n_mix_rot = nodes.new('ShaderNodeMix')
    n_mix_rot.data_type = 'VECTOR'
    n_mix_rot.location = (600, -50)
    links.new(n_in.outputs["Use Normal Rotation"], n_mix_rot.inputs["Factor"])
    links.new(n_sub_offset.outputs["Vector"], n_mix_rot.inputs["A"])
    links.new(n_rot_vec.outputs["Vector"], n_mix_rot.inputs["B"])

    # Deformed Position = Editable_Vertex_Pos + Offset
    n_add_final = nodes.new('ShaderNodeVectorMath')
    n_add_final.operation = 'ADD'
    n_add_final.location = (740, -50)
    links.new(n_edit_vpos.outputs["Value"], n_add_final.inputs[0])
    links.new(n_mix_rot.outputs["Result"], n_add_final.inputs[1])

    # Blend by overall Influence
    n_mix_inf = nodes.new('ShaderNodeMix')
    n_mix_inf.data_type = 'VECTOR'
    n_mix_inf.location = (740, 150)
    links.new(n_in.outputs["Influence"], n_mix_inf.inputs["Factor"])
    links.new(n_pos.outputs["Position"], n_mix_inf.inputs["A"])
    links.new(n_add_final.outputs["Vector"], n_mix_inf.inputs["B"])

    # Set Final Position
    n_set_pos = nodes.new('GeometryNodeSetPosition')
    n_set_pos.location = (860, 50)
    links.new(n_in.outputs["Geometry"], n_set_pos.inputs["Geometry"])
    links.new(n_mix_inf.outputs["Result"], n_set_pos.inputs["Position"])
    links.new(n_set_pos.outputs["Geometry"], n_out.inputs["Geometry"])

    return ng


# ------------------------------------------------------------------------
# Properties & Operators
# ------------------------------------------------------------------------
class PCPROXY_Properties(bpy.types.PropertyGroup):
    point_cloud_obj: bpy.props.PointerProperty(
        name="Point Cloud",
        type=bpy.types.Object,
        description="Select your Point Cloud or Splat Mesh object",
        poll=lambda self, obj: obj.type in {'POINTCLOUD', 'MESH'},
    )
    editable_proxy_obj: bpy.props.PointerProperty(
        name="Editable Proxy",
        type=bpy.types.Object,
        description="The generated Editable Vertex Proxy Mesh",
        poll=lambda self, obj: obj.type == 'MESH',
    )
    armature_obj: bpy.props.PointerProperty(
        name="Armature (Optional)",
        type=bpy.types.Object,
        description="Optional Armature to bind the Editable Vertex Proxy to",
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    proxy_type: bpy.props.EnumProperty(
        name="Proxy Shape",
        items=[
            ('HULL', "Convex Hull Cage", "Fits a tight subdivided convex hull cage around the point cloud"),
            ('BOX', "Subdivided Grid Box", "Creates a clean subdivided 3D lattice box around the point cloud"),
            ('SUBSAMPLED_VERTS', "Subsampled Cloud Verts", "Samples vertices directly from the point cloud positions"),
        ],
        default='HULL',
    )
    subdivisions: bpy.props.IntProperty(
        name="Proxy Resolution",
        description="Subdivision cuts on the generated vertex proxy",
        default=2,
        min=0,
        max=5,
    )
    cage_padding: bpy.props.FloatProperty(
        name="Cage Offset /Merge Dist",
        description="Outward padding or minimum distance between proxy vertices",
        default=0.05,
        min=0.001,
        max=2.0,
    )


class PCPROXY_OT_GenerateBindProxy(bpy.types.Operator):
    """Generate an Editable Vertex Proxy from the Point Cloud and bind them via Geometry Nodes"""
    bl_idname = "pcproxy.generate_and_bind"
    bl_label = "1. Create & Bind Editable Vertex Proxy"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcproxy_props
        pc_obj = props.point_cloud_obj

        if not pc_obj:
            self.report({'ERROR'}, "Please select a Point Cloud object first.")
            return {'CANCELLED'}

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Remove old proxies if re-generating
        edit_name = f"{pc_obj.name}_EditableProxy"
        rest_name = f"{pc_obj.name}_RestProxy_Hidden"
        for old_name in (edit_name, rest_name):
            if old_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[old_name], do_unlink=True)

        # 1. Generate Editable Proxy Mesh
        proxy_mesh = generate_editable_proxy_mesh(
            pc_obj,
            proxy_type=props.proxy_type,
            subdivisions=props.subdivisions,
            padding=props.cage_padding,
        )

        edit_proxy = bpy.data.objects.new(edit_name, proxy_mesh)
        context.collection.objects.link(edit_proxy)
        edit_proxy.matrix_world = pc_obj.matrix_world.copy()
        edit_proxy.display_type = 'WIRE'
        edit_proxy.show_in_front = True
        edit_proxy.hide_render = True

        # 2. Generate Hidden Static Rest Proxy (identical vertex count & indices)
        rest_mesh = proxy_mesh.copy()
        rest_proxy = bpy.data.objects.new(rest_name, rest_mesh)
        context.collection.objects.link(rest_proxy)
        rest_proxy.matrix_world = pc_obj.matrix_world.copy()
        rest_proxy.parent = edit_proxy
        rest_proxy.matrix_parent_inverse = edit_proxy.matrix_world.inverted()
        rest_proxy.hide_render = True
        rest_proxy.hide_set(True)

        # Store pointer to Editable Proxy in UI
        props.editable_proxy_obj = edit_proxy

        # 3. Create & Attach Geometry Nodes Modifier to Point Cloud
        ng = build_vertex_proxy_nodetree("GN_PointCloud_VertexProxy")
        mod_name = "PC_VertexProxy_GN"
        mod = pc_obj.modifiers.get(mod_name)
        if not mod:
            mod = pc_obj.modifiers.new(name=mod_name, type='NODES')
        mod.node_group = ng

        for item in ng.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                if item.name == "Rest Proxy":
                    mod[item.identifier] = rest_proxy
                elif item.name == "Editable Proxy":
                    mod[item.identifier] = edit_proxy
                elif item.name == "Influence":
                    mod[item.identifier] = 1.0
                elif item.name == "Use Normal Rotation":
                    mod[item.identifier] = 1.0

        # Select the Editable Proxy and switch to Edit Mode so user can immediately move vertices
        bpy.ops.object.select_all(action='DESELECT')
        edit_proxy.select_set(True)
        context.view_layer.objects.active = edit_proxy

        self.report({'INFO'}, f"Created '{edit_proxy.name}'. Move its vertices in Edit/Sculpt mode to deform '{pc_obj.name}'!")
        return {'FINISHED'}


class PCPROXY_OT_RebindAfterTopologyEdit(bpy.types.Operator):
    """Update Rest Proxy after adding/deleting vertices on the Editable Proxy"""
    bl_idname = "pcproxy.rebind_rest_state"
    bl_label = "Update Rest State (After Adding/Deleting Verts)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcproxy_props
        pc_obj = props.point_cloud_obj
        edit_proxy = props.editable_proxy_obj

        if not pc_obj or not edit_proxy:
            self.report({'ERROR'}, "Both Point Cloud and Editable Proxy must be set.")
            return {'CANCELLED'}

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        rest_name = f"{pc_obj.name}_RestProxy_Hidden"
        rest_proxy = bpy.data.objects.get(rest_name)
        if not rest_proxy:
            rest_proxy = bpy.data.objects.new(rest_name, edit_proxy.data.copy())
            context.collection.objects.link(rest_proxy)
        else:
            rest_proxy.data = edit_proxy.data.copy()

        rest_proxy.matrix_world = edit_proxy.matrix_world.copy()
        rest_proxy.parent = edit_proxy
        rest_proxy.matrix_parent_inverse = edit_proxy.matrix_world.inverted()
        rest_proxy.hide_render = True
        rest_proxy.hide_set(True)

        self.report({'INFO'}, "Updated Rest State to match current Editable Proxy topology!")
        return {'FINISHED'}


class PCPROXY_OT_RigProxyToArmature(bpy.types.Operator):
    """Parent the Editable Vertex Proxy to the selected Armature with Automatic Weights"""
    bl_idname = "pcproxy.rig_proxy_to_armature"
    bl_label = "2. Bind Editable Proxy to Armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcproxy_props
        edit_proxy = props.editable_proxy_obj
        arm_obj = props.armature_obj

        if not edit_proxy or not arm_obj:
            self.report({'ERROR'}, "Select both an Editable Proxy and an Armature.")
            return {'CANCELLED'}

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        edit_proxy.select_set(True)
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        self.report({'INFO'}, f"Bound '{edit_proxy.name}' to '{arm_obj.name}'! You can now pose bones OR edit proxy vertices.")
        return {'FINISHED'}


class PCPROXY_OT_RemoveProxySetup(bpy.types.Operator):
    """Remove the Geometry Nodes modifier and delete proxy meshes"""
    bl_idname = "pcproxy.remove_setup"
    bl_label = "Remove Proxy & Unbind"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcproxy_props
        pc_obj = props.point_cloud_obj

        if not pc_obj:
            self.report({'ERROR'}, "Select the Point Cloud object first.")
            return {'CANCELLED'}

        mod = pc_obj.modifiers.get("PC_VertexProxy_GN")
        if mod:
            pc_obj.modifiers.remove(mod)

        for suffix in ("_EditableProxy", "_RestProxy_Hidden"):
            name = f"{pc_obj.name}{suffix}"
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

        props.editable_proxy_obj = None
        self.report({'INFO'}, "Removed Vertex Proxy setup.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
# UI Panel in 3D Viewport Sidebar (N-Panel)
# ------------------------------------------------------------------------
class PCPROXY_PT_MainPanel(bpy.types.Panel):
    bl_label = "PC Editable Vertex Proxy (5.3)"
    bl_idname = "PCPROXY_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PC Proxy"

    def draw(self, context):
        layout = self.layout
        props = context.scene.pcproxy_props

        col = layout.column(align=True)
        col.label(text="Target Objects:")
        col.prop(props, "point_cloud_obj")
        col.prop(props, "editable_proxy_obj")

        layout.separator()

        box = layout.box()
        box.label(text="Vertex Proxy Generation:")
        box.prop(props, "proxy_type")
        box.prop(props, "subdivisions")
        box.prop(props, "cage_padding")

        col_gen = box.column(align=True)
        col_gen.scale_y = 1.4
        col_gen.operator("pcproxy.generate_and_bind")
        box.operator("pcproxy.rebind_rest_state")

        layout.separator()

        box_arm = layout.box()
        box_arm.label(text="Optional Armature Rigging:")
        box_arm.prop(props, "armature_obj")
        col_arm = box_arm.column(align=True)
        col_arm.scale_y = 1.3
        col_arm.operator("pcproxy.rig_proxy_to_armature")

        layout.separator()
        layout.operator("pcproxy.remove_setup")


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------
classes = (
    PCPROXY_Properties,
    PCPROXY_OT_GenerateBindProxy,
    PCPROXY_OT_RebindAfterTopologyEdit,
    PCPROXY_OT_RigProxyToArmature,
    PCPROXY_OT_RemoveProxySetup,
    PCPROXY_PT_MainPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pcproxy_props = bpy.props.PointerProperty(type=PCPROXY_Properties)


def unregister():
    if hasattr(bpy.types.Scene, "pcproxy_props"):
        del bpy.types.Scene.pcproxy_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()