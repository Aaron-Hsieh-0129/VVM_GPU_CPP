# Video gallery

## Dynamics benchmarks (regression-aligned)

These cases match the **idealized** modes and **automated regression** configs in [`tests/configs/`](idealized.md#automated-regression-tests). They isolate the dynamical core (no full topography or land physics in those tests).

| Case | Config / mode | Embed |
| ---- | ------------- | ----- |
| Advection u | `advection_u` | below |
| Advection v | `advection_v` | below |
| Advection w | `advection_w` | below |
| Stretching | `stretching` | below |
| Twisting | `twisting` | below |
| 2D bubble | `2dbubble` (e.g. `tests/configs/2dbubble.json`) | below |
| 3D bubble | Warm-bubble dynamics in 3D visualization (same idealized family as 2D bubble; see [Idealized simulations](idealized.md)) | below |
| TaiwanVVM | TaiwanVVM -- topography, land, and physics | below |   
| RCE self-aggregation (SST 300 K) | RCE self-aggregation (SST 300 K) 100 days | below |   
| RCE self-aggregation (RCEMIP, SST 305 K) | RCE self-aggregation (RCEMIP, SST 305 K) 200 days | below |   

### Advection u

<!-- <iframe width="100%" height="400" src="https://www.youtube.com/embed/7qR-_FeTCc8v" title="GVVM advection_u benchmark" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> -->
<iframe width="100%" height="600" src="https://www.youtube.com/embed/7qR-_FeTCc8" title="GVVM advection u" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Advection v

<iframe width="100%" height="600" src="https://www.youtube.com/embed/4q6sFE4QlrA" title="GVVM advection v" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Advection w

<iframe width="100%" height="600" src="https://www.youtube.com/embed/zaMPpXLgrQY" title="GVVM advection w" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Stretching

<iframe width="100%" height="600" src="https://www.youtube.com/embed/Feoa3RW3als" title="GVVM stretching term" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Twisting

<iframe width="100%" height="600" src="https://www.youtube.com/embed/ZtVoWcYPmtA" title="GVVM twisting" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 2D bubble (no physics)

<iframe width="100%" height="600" src="https://www.youtube.com/embed/zmWaabfmZ8Y" title="GVVM 2dbubble" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 3D bubble (no physics)

<iframe width="100%" height="600" src="https://www.youtube.com/embed/gSi1ukT0mXw" title="GVVM 3dbubble" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Full physics and complex domains

### TaiwanVVM — topography, land, and physics

<iframe width="100%" height="600" src="https://www.youtube.com/embed/6wFKTAkBJKw" title="GTaiwanVVM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### RCE self-aggregation (SST 300 K)

<iframe width="100%" height="600" src="https://www.youtube.com/embed/QMaB25El1H4" title="GVVM RCE 300K" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### RCE self-aggregation (RCEMIP, SST 305 K)

<iframe width="100%" height="600" src="https://www.youtube.com/embed/w2JkpAE6j8U" title="GVVM rcemip 305K" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


## See also

- [Idealized simulations](idealized.md) — `idealized_test` modes and regression tests  
- [TaiwanVVM](taiwan-vvm.md) — NetCDF topography and land-surface workflow  
- [Quick Start](../quick-start.md) — build and run
