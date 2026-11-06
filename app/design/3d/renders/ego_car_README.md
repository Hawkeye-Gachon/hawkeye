# Ego car renders (`ego_car_*.webp`)

**What it is.** This folder documents the "my car" image in the left panel of the driving screen. The car is seen from behind and slightly above, like Tesla's driving visualisation in its light theme. It is a **generic modern electric fastback sedan, built procedurally in Blender. It is not a model of any real car or brand, and it has no logos, badges or text.** The paint is pearl white, the canopy and roof are tinted glass, there is a full-width rear light bar, and the wheels are black with aero covers. No stock assets and no generative models were used.

**Outputs.** The files go to `app/src/main/res/drawable-nodpi/`. Each is 1200×1000 RGBA WebP (q90) with a transparent background, a soft contact shadow and a state-tinted ground glow:

| file | state | accent |
|---|---|---|
| `ego_car_observing.webp` | normal | cool white `#9DB4D8` (faint) |
| `ego_car_caution.webp` | attention | amber `#F5A524` |
| `ego_car_rest.webp` | rest recommended | red `#E5484D` |
| `ego_car_unavailable.webp` | analysis unavailable | gray, car desaturated and dimmed, light bar off |
| `ego_car_paused.webp` | resting / paused | violet `#6E6AE8` |

The accent appears in three places: the rear light bar, a low rim light on the lower body, and a soft pool of light on the ground around and behind the car.

**How to re-render** (Blender 5.2, about 5 min on an Apple M5):

```sh
/opt/homebrew/bin/blender -b --factory-startup -P design/3d/ego_car.py -- --samples 128
# options: --states caution,rest   --scale 0.5 (drafts)   --preview-dir DIR (composites on #EEF0F2)
#          --view side|top|rear34 (debug views)   --out-dir DIR   --cpu
```

**How it works.** `design/3d/ego_car.py` is the only source:

1. **Geometry.** The body is lofted from cross-sections (Catmull-Clark subdivision). Booleans cut the wheel arches, and the tyres and covers are revolved profiles. The light bar is placed by ray-casting onto the rear face.
2. **Glass.** The glass edges come from a per-vertex field read by the shader, so the outline stays smooth.
3. **Rendering.** Each state gets a Cycles beauty pass with `film_transparent` and a shadow-catcher ground. A low-sample pass with the ground hidden gives the car's coverage mask.
4. **Post-processing.** The same file also runs under the system `python3` with numpy and Pillow, and Blender calls it automatically. It layers shadow, glow and car in sRGB. The glow is computed on the ground plane from the exported camera, and both shadow and glow fade out before the image edges.

Look parameters (camera, exposure, accents, glow strength) are constants at the top of the script.
