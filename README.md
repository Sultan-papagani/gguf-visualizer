# GGUF VIS — 3D LLM Model Visualizer

A browser-based tool that renders `.gguf` language model files as interactive 3D point clouds. Instead of reading a model's name to understand it, you can **see** its architecture — dense vs MoE, attention head count, layer depth, FFN size, and more — all encoded into a spatial structure you can fly through.

try it here: https://sultan-papagani.github.io/gguf-visualizer/

![No dependencies to install. Just open and drop a file.](https://img.shields.io/badge/zero--install-just_open_index.html-6c8aff)

## What it does

- **Parses `.gguf` files entirely in the browser** — no server, no uploads, your model files never leave your machine.
- **Renders 1M–6M points** as a 3D point cloud using Three.js with custom shaders, where each point represents a decimated sample of the model's parameters.
- **Encodes architecture into shape**: attention Q/K/V heads are laid out by count, FFN blocks scale with hidden size, MoE experts get their own columns, and layers stack along the Z axis. Different models produce visually distinct silhouettes.
- **Neural connection lines** draw data-flow pathways between tensor regions (embedding → attention → FFN → next layer → output), like synapses between neurons.
- **FPS-style free-roam camera** — fly through the model with WASD + mouse look.

## Quick start

No build step. No npm install. Just serve the folder:

```bash
npx serve -l 3000
```

Then open `http://localhost:3000` and drag-drop any `.gguf` file onto the page.

Or simply open `index.html` directly in a browser (the Web Worker fallback handles `file://` gracefully).

## Controls

| Input | Action |
|---|---|
| **Click canvas** | Lock cursor for free-roam |
| **W A S D** | Move forward / left / back / right |
| **Space** | Move up |
| **Ctrl / C** | Move down |
| **Shift** | Sprint (3× speed) |
| **Scroll wheel** | Adjust movement speed |
| **Esc** | Release cursor |
| **Tab** | Toggle sidebar |

## Sidebar controls

| Control | What it does |
|---|---|
| **Points** slider | Number of rendered points (500K – 6M) |
| **Point Size** slider | Dot size (0.1 – 5.0, default 0.6) |
| **Color** dropdown | `Layer Depth` (default), `Tensor Type`, or `Weight Value` |
| **Connections** checkbox | Toggle neural pathway lines between tensor regions |
| **Density** slider | Number of connection lines (0.1× – 3.0×, appears when connections are on) |

## Color modes

- **Layer Depth** — green → blue → purple gradient from layer 0 to layer N. Global tensors (embedding, output) are gray.
- **Tensor Type** — each tensor category gets a distinct color: blue for attention, orange for FFN, pink for MoE experts, green for embedding, purple for output, yellow for norms.
- **Weight Value** — diverging blue → white → red colormap based on actual dequantized weight values sampled from the file.

## Architecture recognition

The 3D layout is designed so you can visually identify model characteristics:

| Feature | What you see |
|---|---|
| **Layer count** | Depth of the structure along the Z axis |
| **Attention heads** | Width of the Q/K/V blocks (more heads = wider) |
| **GQA (grouped query attention)** | K/V blocks are shorter and narrower than Q |
| **FFN hidden size** | Height of the FFN gate/up/down columns |
| **MoE (mixture of experts)** | Multiple FFN columns per layer instead of one |
| **Expert count** | Number of columns in the MoE section |
| **Model size** | Overall volume and density of the point cloud |
| **Quantization** | Weight Value color mode shows quantization artifacts |

## Project structure

```
visuals/
├── index.html          # Single-page app: HTML + CSS + UI
└── js/
    ├── app.js           # Main orchestrator: file upload → parse → render
    ├── gguf-parser.js   # Binary GGUF v2/v3 parser + weight sampling
    ├── parse-worker.js  # Web Worker for background header parsing
    ├── point-cloud.js   # 3D layout engine + neural connection generator
    └── renderer.js      # Three.js scene, shaders, FPS camera
```

## Supported formats

- **GGUF v2 and v3** files (the format used by llama.cpp, ollama, LM Studio, etc.)
- Quantization types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K through Q6_K, Q8_K, and IQ types
- Tested with LLaMA, Mistral, Mixtral, Qwen, Phi, Gemma, and other architectures

## Performance notes

- **Header parsing** runs in a Web Worker to keep the UI responsive (falls back to main thread if workers are unavailable).
- **Weight sampling** for the Weight Value color mode uses parallel batched file reads (8 tensors at a time via `Promise.all`).
- **Rendering** uses custom GLSL shaders with additive blending, distance-based point sizing, and exponential fog for depth perception.
- The Layer Depth and Tensor Type color modes skip file I/O entirely (no weight sampling needed), making them near-instant.

## Dependencies

Zero runtime dependencies to install. Three.js is loaded from CDN:

```
three@0.163.0 — https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js
```

## License

Do whatever you want with it.


