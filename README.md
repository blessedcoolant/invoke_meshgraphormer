# Hand Refiner w/ MeshGraphormer for InvokeAI

A quick implementation of hand refining with Microsoft's Meshgraphormer for InvokeAI.

Go to your `nodes` folder in your `root` directory and clone this repo

```
git clone https://github.com/blessedcoolant/invoke_meshgraphormer.git
```

You might have to install a couple of extra dependencies in your Invoke `venv`.

```
pip install trimesh rtree yacs
```

Example workflow is provided in the `workflow` folder.

This extension consists of on node - `Hand Depth w/ MeshGraphormer`. This node takes in an image and outputs a hand depth map and a mask for the hand area. Use this depth map with a depth ControlNet model and you can use the mask to create a `Denoise Mask` for your hand-fix pass so only the hand are denoised.

Feel free to fix anything. I'm sure there's a bunch of issues.
