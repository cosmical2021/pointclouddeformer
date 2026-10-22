bl_info = {
    "name": "Point Cloud Geometry Nodes Rigger",
    "author": "Gemini",
    "version": (1, 3, 0),
    "blender": (5, 3, 0),
    "location": "View3D > Sidebar (N) > PC Rig",
    "description": "Rigs Point Cloud and Vertex-Cloud/Splat objects to Armatures using Geometry Nodes",
    "category": "Rigging",
}

import bpy
import bmesh
from mathutils import Vector, Matrix


# ------------------------------------------------------------------------
# Core Helper: Generate Bone Envelope Proxy Mesh for Automatic Weights
# ------------------------------------------------------------------------
def create_bone_hull_mesh(context, armature_obj, padding=0.08, subdivisions=2):
    """Creates a closed volumetric mesh around all deform bones for heat-weight solving."""
    bm = bmesh.new()

    arm_mw = armature_obj.matrix_world
    deform_bones = [b for b in armature_obj.data.bones if b.use_deform]
    if not deform_bones:
        deform_bones = list(armature_obj.data.bones)

    for bone in deform_bones:
        head_w = arm_mw @ bone.head_local
        tail_w = arm_mw @ bone.tail_local
        bone_vec = tail_w - head_w
        length = max(bone_vec.length, 0.001)

        radius = max(bone.head_radius, bone.tail_radius, length * 0.15, padding)

        z_axis = bone_vec.normalized()
        up = Vector((0, 0, 1)) if abs(z_axis.z) < 0.99 else Vector((1, 0, 0))
        x_axis = up.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()

        rot_mat = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
        mid_w = (head_w + tail_w) * 0.5
        loc_mat = Matrix.Translation(mid_w)
        scale_mat = Matrix.Diagonal((radius, radius, length * 0.55, 1.0))

        xform = loc_mat @ rot_mat @ scale_mat
        bmesh.ops.create_cube(bm, size=2.0, matrix=xform)

    if subdivisions > 0:
        bmesh.ops.subdivide_edges(
            bm,
            edges=bm.edges[:],
            cuts=subdivisions,
            use_grid_fill=True,
        )

    mesh_data = bpy.data.meshes.new(f"{armature_obj.name}_PC_ProxyMesh")
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()
    return mesh_data


# ------------------------------------------------------------------------
# Core Helper: Build the Geometry Nodes Point Cloud Skinning Tree
# ------------------------------------------------------------------------
def build_pointcloud_skinning_nodetree(node_group_name):
    """
    Constructs a Geometry Nodes tree that skins a Point Cloud or Splat Mesh
    using Rest & Deformed proxy meshes via Sample Nearest + Normal Rotation Delta.
    """
    if node_group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[node_group_name])

    ng = bpy.data.node_groups.new(name=node_group_name, type='GeometryNodeTree')
    ng.is_modifier = True

    # 1. Define Interface Sockets (Blender 5.x NodeTreeInterface API)
    iface = ng.interface
    iface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    iface.new_socket(name="Rest Proxy", in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(name="Deformed Proxy", in_out='INPUT', socket_type='NodeSocketObject')
    in_influence = iface.new_socket(name="Bind Influence", in_out='INPUT', socket_type='NodeSocketFloat')
    in_influence.default_value = 1.0
    in_influence.min_value = 0.0
    in_influence.max_value = 1.0

    nodes = ng.nodes
    links = ng.links

    # Group Input / Output
    n_in = nodes.new('NodeGroupInput')
    n_in.location = (-900, 0)
    n_out = nodes.new('NodeGroupOutput')
    n_out.location = (900, 0)

    # Object Info: Rest Proxy & Deformed Proxy (Relative Space)
    n_rest_info = nodes.new('GeometryNodeObjectInfo')
    n_rest_info.transform_space = 'RELATIVE'
    n_rest_info.location = (-650, -150)

    n_def_info = nodes.new('GeometryNodeObjectInfo')
    n_def_info.transform_space = 'RELATIVE'
    n_def_info.location = (-650, -400)

    links.new(n_in.outputs["Rest Proxy"], n_rest_info.inputs["Object"])
    links.new(n_in.outputs["Deformed Proxy"], n_def_info.inputs["Object"])

    # Field Inputs (evaluated on the Proxy Meshes during Sample Index)
    n_pos = nodes.new('GeometryNodeInputPosition')
    n_pos.location = (-650, 150)

    n_norm = nodes.new('GeometryNodeInputNormal')
    n_norm.location = (-650, 50)

    # Sample Nearest Vertex on Rest Proxy
    n_nearest_idx = nodes.new('GeometryNodeSampleNearest')
    n_nearest_idx.domain = 'POINT'
    n_nearest_idx.location = (-380, -200)
    links.new(n_rest_info.outputs["Geometry"], n_nearest_idx.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_nearest_idx.inputs["Sample Position"])

    # Sample Rest Proxy Position & Normal at Nearest Index
    n_rest_vpos = nodes.new('GeometryNodeSampleIndex')
    n_rest_vpos.data_type = 'FLOAT_VECTOR'
    n_rest_vpos.domain = 'POINT'
    n_rest_vpos.location = (-100, -100)
    links.new(n_rest_info.outputs["Geometry"], n_rest_vpos.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_rest_vpos.inputs["Value"])
    links.new(n_nearest_idx.outputs["Index"], n_rest_vpos.inputs["Index"])

    n_def_vpos = nodes.new('GeometryNodeSampleIndex')
    n_def_vpos.data_type = 'FLOAT_VECTOR'
    n_def_vpos.domain = 'POINT'
    n_def_vpos.location = (-100, -300)
    links.new(n_def_info.outputs["Geometry"], n_def_vpos.inputs["Geometry"])
    links.new(n_pos.outputs["Position"], n_def_vpos.inputs["Value"])
    links.new(n_nearest_idx.outputs["Index"], n_def_vpos.inputs["Index"])

    n_rest_vnorm = nodes.new('GeometryNodeSampleIndex')
    n_rest_vnorm.data_type = 'FLOAT_VECTOR'
    n_rest_vnorm.domain = 'POINT'
    n_rest_vnorm.location = (-100, -500)
    links.new(n_rest_info.outputs["Geometry"], n_rest_vnorm.inputs["Geometry"])
    links.new(n_norm.outputs["Normal"], n_rest_vnorm.inputs["Value"])
    links.new(n_nearest_idx.outputs["Index"], n_rest_vnorm.inputs["Index"])

    n_def_vnorm = nodes.new('GeometryNodeSampleIndex')
    n_def_vnorm.data_type = 'FLOAT_VECTOR'
    n_def_vnorm.domain = 'POINT'
    n_def_vnorm.location = (-100, -700)
    links.new(n_def_info.outputs["Geometry"], n_def_vnorm.inputs["Geometry"])
    links.new(n_norm.outputs["Normal"], n_def_vnorm.inputs["Value"])
    links.new(n_nearest_idx.outputs["Index"], n_def_vnorm.inputs["Index"])

    # Compute Offset Vector = Point_Pos - Rest_Vertex_Pos
    n_sub_offset = nodes.new('ShaderNodeVectorMath')
    n_sub_offset.operation = 'SUBTRACT'
    n_sub_offset.location = (150, 50)
    links.new(n_pos.outputs["Position"], n_sub_offset.inputs[0])
    links.new(n_rest_vpos.outputs["Value"], n_sub_offset.inputs[1])

    # Align Rotation from Rest Normal to Deformed Normal
    n_align_rot = nodes.new('FunctionNodeAlignRotationToVector')
    n_align_rot.location = (150, -450)
    links.new(n_def_vnorm.outputs["Value"], n_align_rot.inputs["Vector"])

    n_rot_rest = nodes.new('FunctionNodeAlignRotationToVector')
    n_rot_rest.location = (150, -650)
    links.new(n_rest_vnorm.outputs["Value"], n_rot_rest.inputs["Vector"])

    n_inv_rot = nodes.new('FunctionNodeInvertRotation')
    n_inv_rot.location = (320, -650)
    links.new(n_rot_rest.outputs["Rotation"], n_inv_rot.inputs["Rotation"])

    n_rot_delta = nodes.new('FunctionNodeRotateRotation')
    n_rot_delta.location = (480, -500)
    links.new(n_inv_rot.outputs["Rotation"], n_rot_delta.inputs["Rotation"])
    links.new(n_align_rot.outputs["Rotation"], n_rot_delta.inputs["Rotate By"])

    # Rotate the Point Offset by the Normal Rotation Delta
    n_rot_vec = nodes.new('FunctionNodeRotateVector')
    n_rot_vec.location = (480, -150)
    links.new(n_sub_offset.outputs["Vector"], n_rot_vec.inputs["Vector"])
    links.new(n_rot_delta.outputs["Rotation"], n_rot_vec.inputs["Rotation"])

    # Final Target Position = Deformed_Vertex_Pos + Rotated_Offset
    n_add_final = nodes.new('ShaderNodeVectorMath')
    n_add_final.operation = 'ADD'
    n_add_final.location = (650, -100)
    links.new(n_def_vpos.outputs["Value"], n_add_final.inputs[0])
    links.new(n_rot_vec.outputs["Vector"], n_add_final.inputs[1])

    # Mix between Original Position and Rigged Position by Influence
    n_mix = nodes.new('ShaderNodeMix')
    n_mix.data_type = 'VECTOR'
    n_mix.location = (650, 150)
    links.new(n_in.outputs["Bind Influence"], n_mix.inputs["Factor"])
    links.new(n_pos.outputs["Position"], n_mix.inputs["A"])
    links.new(n_add_final.outputs["Vector"], n_mix.inputs["B"])

    # Set Position on Point Cloud / Splat
    n_set_pos = nodes.new('GeometryNodeSetPosition')
    n_set_pos.location = (780, 50)
    links.new(n_in.outputs["Geometry"], n_set_pos.inputs["Geometry"])
    links.new(n_mix.outputs["Result"], n_set_pos.inputs["Position"])
    links.new(n_set_pos.outputs["Geometry"], n_out.inputs["Geometry"])

    return ng


# ------------------------------------------------------------------------
# Properties & Operators
# ------------------------------------------------------------------------
class PCRIG_Properties(bpy.types.PropertyGroup):
    point_cloud_obj: bpy.props.PointerProperty(
        name="Point Cloud",
        type=bpy.types.Object,
        description="Select a Point Cloud or Vertex-Cloud/Splat Mesh object",
        poll=lambda self, obj: obj.type in {'POINTCLOUD', 'MESH'},
    )
    armature_obj: bpy.props.PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        description="Select the target Armature to rig the Point Cloud to",
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    proxy_subdivisions: bpy.props.IntProperty(
        name="Weight Resolution",
        description="Subdivision level of the internal weight proxy (higher = smoother deformation)",
        default=3,
        min=1,
        max=5,
    )
    envelope_padding: bpy.props.FloatProperty(
        name="Envelope Padding",
        description="Radius padding around bones for automatic weight transfer",
        default=0.10,
        min=0.01,
        max=2.0,
    )


class PCRIG_OT_BindPointCloud(bpy.types.Operator):
    """Bind the selected Point Cloud to the Armature using Geometry Nodes"""
    bl_idname = "pcrig.bind_point_cloud"
    bl_label = "Bind Point Cloud to Armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcrig_props
        pc_obj = props.point_cloud_obj
        arm_obj = props.armature_obj

        if not pc_obj or not arm_obj:
            self.report({'ERROR'}, "Please select both a Point Cloud object and an Armature.")
            return {'CANCELLED'}

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        col_name = "PC_Rig_Proxies_DoNotDelete"
        if col_name not in bpy.data.collections:
            proxy_col = bpy.data.collections.new(col_name)
            context.scene.collection.children.link(proxy_col)
        else:
            proxy_col = bpy.data.collections[col_name]

        rest_name = f"{pc_obj.name}_RestProxy"
        def_name = f"{pc_obj.name}_DeformProxy"
        for old_name in (rest_name, def_name):
            if old_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[old_name], do_unlink=True)

        # 1. Generate Proxy Hull Mesh
        hull_mesh = create_bone_hull_mesh(
            context,
            arm_obj,
            padding=props.envelope_padding,
            subdivisions=props.proxy_subdivisions,
        )

        # 2. Create Deformed Proxy & bind with Automatic Weights
        def_proxy = bpy.data.objects.new(def_name, hull_mesh)
        proxy_col.objects.link(def_proxy)

        bpy.ops.object.select_all(action='DESELECT')
        def_proxy.select_set(True)
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        # 3. Create Static Rest Proxy
        rest_mesh = hull_mesh.copy()
        rest_proxy = bpy.data.objects.new(rest_name, rest_mesh)
        proxy_col.objects.link(rest_proxy)
        rest_proxy.parent = arm_obj
        rest_proxy.matrix_parent_inverse = arm_obj.matrix_world.inverted()

        for p_obj in (def_proxy, rest_proxy):
            p_obj.hide_render = True
            p_obj.display_type = 'WIRE'
            p_obj.hide_set(True)

        # 4. Build Geometry Nodes Modifier on the Point Cloud
        ng_name = "GN_PointCloud_Armature_Skin"
        ng = build_pointcloud_skinning_nodetree(ng_name)

        mod_name = "PointCloud_Armature_GN"
        mod = pc_obj.modifiers.get(mod_name)
        if not mod:
            mod = pc_obj.modifiers.new(name=mod_name, type='NODES')
        mod.node_group = ng

        # Assign Socket Inputs via Identifier map
        for item in ng.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                if item.name == "Rest Proxy":
                    mod[item.identifier] = rest_proxy
                elif item.name == "Deformed Proxy":
                    mod[item.identifier] = def_proxy
                elif item.name == "Bind Influence":
                    mod[item.identifier] = 1.0

        pc_obj.parent = arm_obj
        pc_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()
        pc_obj.update_tag()

        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        self.report({'INFO'}, f"Successfully rigged '{pc_obj.name}' to '{arm_obj.name}'!")
        return {'FINISHED'}


class PCRIG_OT_UnbindPointCloud(bpy.types.Operator):
    """Remove the Geometry Nodes rig and proxy objects from the Point Cloud"""
    bl_idname = "pcrig.unbind_point_cloud"
    bl_label = "Unbind & Clean Proxies"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pcrig_props
        pc_obj = props.point_cloud_obj

        if not pc_obj:
            self.report({'ERROR'}, "Select the Point Cloud object first.")
            return {'CANCELLED'}

        mod = pc_obj.modifiers.get("PointCloud_Armature_GN")
        if mod:
            pc_obj.modifiers.remove(mod)

        for suffix in ("_RestProxy", "_DeformProxy"):
            obj_name = f"{pc_obj.name}{suffix}"
            if obj_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

        self.report({'INFO'}, f"Removed Point Cloud rig from '{pc_obj.name}'.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
# UI Panel in 3D Viewport Sidebar (N-Panel) - Blender 5.3 Safe Icons
# ------------------------------------------------------------------------
class PCRIG_PT_MainPanel(bpy.types.Panel):
    bl_label = "Point Cloud GN Rigger (5.3)"
    bl_idname = "PCRIG_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PC Rig"

    def draw(self, context):
        layout = self.layout
        props = context.scene.pcrig_props

        col = layout.column(align=True)
        col.label(text="Target Objects:")
        col.prop(props, "point_cloud_obj")
        col.prop(props, "armature_obj")

        layout.separator()

        box = layout.box()
        box.label(text="Solver Settings:")
        box.prop(props, "proxy_subdivisions")
        box.prop(props, "envelope_padding")

        layout.separator()

        col_ops = layout.column(align=True)
        col_ops.scale_y = 1.4
        col_ops.operator("pcrig.bind_point_cloud", text="Bind Point Cloud to Armature")

        layout.separator()
        layout.operator("pcrig.unbind_point_cloud", text="Unbind & Clean Proxies")


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------
classes = (
    PCRIG_Properties,
    PCRIG_OT_BindPointCloud,
    PCRIG_OT_UnbindPointCloud,
    PCRIG_PT_MainPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pcrig_props = bpy.props.PointerProperty(type=PCRIG_Properties)


def unregister():
    if hasattr(bpy.types.Scene, "pcrig_props"):
        del bpy.types.Scene.pcrig_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()