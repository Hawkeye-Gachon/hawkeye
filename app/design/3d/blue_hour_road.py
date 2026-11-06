"""
DriverGuard 배경 3D 렌더: 블루아워의 고속도로가 지평선으로 사라지는 장면.

실행(헤드리스):
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
      --python design/3d/blue_hour_road.py -- --out out.png --w 2400 --h 1350 --samples 96

앱 상태색(지평선 빛)은 앱이 실시간으로 그리므로, 이 렌더의 지평선은 중립적인 블루아워로 둔다.
"""
import argparse
import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/blue_hour_road.png")
ap.add_argument("--w", type=int, default=960)
ap.add_argument("--h", type=int, default=540)
ap.add_argument("--samples", type=int, default=24)
ap.add_argument("--novol", action="store_true")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--shift", type=float, default=0.10, help="카메라 shift_y: 지평선을 화면 아래쪽으로")
args = ap.parse_args(argv)

# ---------------------------------------------------------------- 초기화
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"


def hex_rgb(h, a=1.0):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


def material(name, base, rough=0.6, metal=0.0, emission=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = hex_rgb(base)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emission:
        bsdf.inputs["Emission Color"].default_value = hex_rgb(emission)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


def box(name, size, loc, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


def plane(name, sx, sy, loc, mat):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (sx, sy, 1)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


FOG_COLOR = "#1d3358"
FOG_DENSITY = 0.0011


def add_fog(mat, density=FOG_DENSITY):
    """카메라 거리에 따라 표면을 안개색으로 섞는다(1 - e^(-d*k)). 볼륨 없이 대기 원근감을 낸다."""
    t = mat.node_tree
    out = t.nodes["Material Output"]
    surf = out.inputs["Surface"].links[0].from_socket
    cam = t.nodes.new("ShaderNodeCameraData")
    mul = t.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = -density
    exp = t.nodes.new("ShaderNodeMath")
    exp.operation = "EXPONENT"
    inv = t.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    emit = t.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = hex_rgb(FOG_COLOR)
    emit.inputs["Strength"].default_value = 1.0
    mix = t.nodes.new("ShaderNodeMixShader")
    t.links.new(cam.outputs["View Distance"], mul.inputs[0])
    t.links.new(mul.outputs["Value"], exp.inputs[0])
    t.links.new(exp.outputs["Value"], inv.inputs[1])
    t.links.new(inv.outputs["Value"], mix.inputs["Fac"])
    t.links.new(surf, mix.inputs[1])
    t.links.new(emit.outputs["Emission"], mix.inputs[2])
    t.links.new(mix.outputs["Shader"], out.inputs["Surface"])


ROAD_LEN = 3000.0
LANE = 3.6
HALF = LANE * 1.5  # 3차선 한쪽 차로

# ---------------------------------------------------------------- 아스팔트(잔골재 노이즈 + 젖은 듯한 거칠기 변화)
asphalt = bpy.data.materials.new("asphalt")
asphalt.use_nodes = True
nt = asphalt.node_tree
bsdf = nt.nodes["Principled BSDF"]
coord = nt.nodes.new("ShaderNodeTexCoord")
noise = nt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 180.0
noise.inputs["Detail"].default_value = 8.0
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = hex_rgb("#141619")
ramp.color_ramp.elements[1].color = hex_rgb("#2a2d33")
nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
rough_noise = nt.nodes.new("ShaderNodeTexNoise")
rough_noise.inputs["Scale"].default_value = 3.0
rough_map = nt.nodes.new("ShaderNodeMapRange")
rough_map.inputs["To Min"].default_value = 0.30
rough_map.inputs["To Max"].default_value = 0.55
nt.links.new(coord.outputs["Object"], rough_noise.inputs["Vector"])
nt.links.new(rough_noise.outputs["Fac"], rough_map.inputs["Value"])
nt.links.new(rough_map.outputs["Result"], bsdf.inputs["Roughness"])
bump = nt.nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.04
nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

# 차선 도료의 유리알 재귀반사를 약한 자체 발광으로 흉내 낸다
paint_white = material("paint_white", "#d9dde4", rough=0.45, emission="#c9d6ee", strength=0.35)
paint_yellow = material("paint_yellow", "#d6a43a", rough=0.45, emission="#c8963a", strength=0.10)
concrete = material("concrete", "#4b4f57", rough=0.85)
steel = material("steel", "#8b929c", rough=0.35, metal=1.0)
ground = material("ground", "#0d1310", rough=0.95)
hill = material("hill", "#0e1726", rough=1.0)

# ---------------------------------------------------------------- 도로·노면 표시
plane("road", HALF * 2 + 1.2, ROAD_LEN, (0, ROAD_LEN / 2 - 20, 0), asphalt)
plane("ground", 4000, ROAD_LEN, (0, ROAD_LEN / 2 - 20, -0.05), ground)

# 차선 점선(10m 칠하고 10m 띄움), 가장자리 실선
for x in (-LANE / 2, LANE / 2):
    y = -10.0
    while y < 900:
        box("dash", (0.15, 10.0, 0.004), (x, y + 5.0, 0.002), paint_white)
        y += 20.0
box("edge_right", (0.18, ROAD_LEN, 0.004), (HALF + 0.15, ROAD_LEN / 2 - 20, 0.002), paint_white)
box("edge_left_a", (0.12, ROAD_LEN, 0.004), (-HALF - 0.15, ROAD_LEN / 2 - 20, 0.002), paint_yellow)
box("edge_left_b", (0.12, ROAD_LEN, 0.004), (-HALF - 0.45, ROAD_LEN / 2 - 20, 0.002), paint_yellow)

# 중앙 분리대(콘크리트 방호벽)
box("median", (0.55, ROAD_LEN, 0.9), (-HALF - 1.1, ROAD_LEN / 2 - 20, 0.45), concrete)

# 오른쪽 가드레일: 레일 + 기둥(4m 간격)
box("rail", (0.06, ROAD_LEN, 0.32), (HALF + 1.2, ROAD_LEN / 2 - 20, 0.62), steel)
bpy.ops.mesh.primitive_cube_add(size=1, location=(HALF + 1.28, 0, 0.4))
post = bpy.context.active_object
post.scale = (0.1, 0.1, 0.8)
bpy.ops.object.transform_apply(scale=True)
post.data.materials.append(steel)
arr = post.modifiers.new("arr", "ARRAY")
arr.count = 220
arr.use_relative_offset = False
arr.use_constant_offset = True
arr.constant_offset_displace = (0, 4.0, 0)

# ---------------------------------------------------------------- 먼 산 능선(노이즈로 변형한 길쭉한 구)
def ridge(name, x, y, sx, sy, sz, seed):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, location=(x, y, 0))
    o = bpy.context.active_object
    o.name = name
    o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    tex = bpy.data.textures.new(name + "_tex", "CLOUDS")
    tex.noise_scale = 0.6
    disp = o.modifiers.new("disp", "DISPLACE")
    disp.texture = tex
    disp.strength = sz * 0.35
    disp.texture_coords = "GLOBAL"
    o.data.materials.append(hill)
    bpy.ops.object.shade_smooth()
    return o


# 낮고 먼 능선: 지평선 빛 띠(앱이 상태색을 그리는 자리)를 가리지 않는다
ridge("ridge_a", -1400, 3600, 1800, 300, 95, 1)
ridge("ridge_b", 900, 4200, 2200, 300, 120, 2)
ridge("ridge_c", 3000, 3400, 1300, 250, 70, 3)
ridge("ridge_d", -3600, 3200, 1400, 250, 60, 4)

for m in bpy.data.materials:
    add_fog(m, FOG_DENSITY * (0.35 if m.name == "hill" else 1.0))

# ---------------------------------------------------------------- 하늘(시선 높이에 따른 그라데이션) + 대기 산란
world = bpy.data.worlds.new("sky")
scene.world = world
world.use_nodes = True
wn = world.node_tree
bg = wn.nodes["Background"]
tc = wn.nodes.new("ShaderNodeTexCoord")
sep = wn.nodes.new("ShaderNodeSeparateXYZ")
sky_ramp = wn.nodes.new("ShaderNodeValToRGB")
els = sky_ramp.color_ramp.elements
els[0].position = 0.0
els[0].color = hex_rgb("#304f80")
els[1].position = 0.45
els[1].color = hex_rgb("#060b16")
mid = els.new(0.05)
mid.color = hex_rgb("#1f3866")
mid2 = els.new(0.16)
mid2.color = hex_rgb("#0f1d38")
wn.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
wn.links.new(sep.outputs["Z"], sky_ramp.inputs["Fac"])
wn.links.new(sky_ramp.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 1.0

vol = wn.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Color"].default_value = hex_rgb("#2b4675")
vol.inputs["Density"].default_value = 0.0002
vol.inputs["Anisotropy"].default_value = 0.2
if not args.novol:
    wn.links.new(vol.outputs["Volume"], wn.nodes["World Output"].inputs["Volume"])

# 낮게 깔린 달빛(노면 광택용)
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
moon = bpy.context.active_object
moon.data.energy = 0.25
moon.data.color = hex_rgb("#9db6e8")[:3]
moon.data.angle = math.radians(2.0)
moon.rotation_euler = (math.radians(-86), 0, math.radians(180))

# ---------------------------------------------------------------- 카메라(운전자 눈높이, 가운데 차로)
bpy.ops.object.camera_add(location=(0.0, -2.0, 1.25))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(90.0), 0, 0)
cam.data.lens = 28
cam.data.shift_y = args.shift
cam.data.clip_end = 6000
scene.camera = cam

# ---------------------------------------------------------------- 렌더
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    if args.cpu:
        raise RuntimeError("cpu requested")
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception:
    scene.cycles.device = "CPU"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.volume_step_rate = 4.0
scene.render.resolution_x = args.w
scene.render.resolution_y = args.h
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Base Contrast"
scene.view_settings.exposure = 0.4
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = args.out
bpy.ops.render.render(write_still=True)
print("RENDERED", args.out)
