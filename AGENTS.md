# AGENTS.md

Guidance for Claude Code (and other AI coding agents) working in this repository.

## Project purpose

This repository is an **educational visualization project** for learning machine learning, deep learning, and reinforcement learning. It is **not** a production library, framework, or research codebase. Its goal is to help a human build intuition about algorithms as quickly as possible through the simplest possible runnable examples and visual output.

Every change you make should serve that goal.

## Core principles

When modifying, adding, or reviewing code in this repo, follow these principles in order:

### 1. Visualization first
Every example must produce a visual artifact — a matplotlib plot, an animation (`.gif`), a rendered image (`.png`), a TensorBoard log, or an interactive display. If an algorithm runs but shows nothing on screen, it does not belong here. Prefer plots that reveal *how* the algorithm works (loss curves, decision boundaries, attention maps, weight evolution, gradient flow, agent trajectories) over plots that only show final results.

### 2. Simplest possible example
Use the smallest dataset, shortest training loop, and fewest layers that still demonstrate the concept. Synthetic data (`np.linspace`, `torch.randn`, toy grids) is preferred over large datasets whenever it still conveys the idea. A reader should be able to run the file and understand the algorithm in minutes, not hours.

### 3. Progressive difficulty (浅入深)
The repo is organized so a learner can move from simple to complex: least squares → neural-network regression → classification → CNN → RNN/LSTM → Seq2Seq → Transformer → RL. When adding a new example, place it at the right level of difficulty and do not pull in advanced machinery that the learner hasn't seen yet at that level.

### 4. Self-contained files
Each example should be a single runnable script (or a small self-contained subfolder). Avoid introducing shared utility modules, abstract base classes, config systems, or framework-style scaffolding. Duplication across examples is fine and often preferred — a learner should be able to read one file top-to-bottom without jumping around.

### 5. Step-by-step visibility
Prefer showing intermediate state over hiding it. Plot loss per epoch, render predictions during training, animate the optimizer's trajectory, print shapes at each layer. The reader should *see* the algorithm take each step, not just its final answer.

### 6. README is the showcase
For every code example, a key code snippet and the generated visualization (image or gif) should appear in `README.md`. When you add a new example:
- Add a new section to `README.md` with the algorithm name.
- Include the most illustrative code fragment (not the full file — just the core idea).
- Embed the generated image/gif with a relative path (e.g. `![](./my_example.png)`).
- Add a link to the new section in the table of contents at the top of `README.md`.

When you modify an existing example in a way that changes its output, regenerate the image/gif and update the README snippet if the key code changed.

## What to do

- Add comments that explain the *why* and the *math*, not the *what*.
- Keep imports minimal and at the top; prefer `numpy`, `matplotlib`, `torch`, `gymnasium` — things a learner likely already knows.
- Use descriptive variable names that match the math (`theta`, `loss`, `grad`, `x_train`).
- When training, use small epoch counts and print/plot progress so the example finishes quickly on a laptop CPU.
- When adding a visualization, make sure the plot has axis labels, a title, and a legend where relevant.

## What to avoid

- Do **not** add production concerns: logging frameworks, CLI argument parsers, config files, Docker, CI pipelines, type-checking scaffolds, test suites (unless a test itself is the teaching artifact).
- Do **not** refactor multiple examples into a shared abstraction "to reduce duplication." Duplication is a feature here.
- Do **not** introduce heavy dependencies for a small gain. If `numpy` + `matplotlib` can show it, use them.
- Do **not** replace simple, explicit loops with clever vectorized one-liners if the loop makes the algorithm clearer to a learner.
- Do **not** remove visualizations, `plt.show()` calls, or saved image outputs — they are the point of the file.
- Do **not** add features the user did not ask for. Scope your change to the educational goal stated in the request.
- Do **not** silently delete existing `.png` / `.gif` artifacts — they are referenced from `README.md`.

## When adding a new algorithm example

Checklist:

1. Create a single self-contained script (e.g. `my_algorithm.py`) at the appropriate difficulty level.
2. Use synthetic or tiny data so it runs in seconds.
3. Produce at least one visualization that reveals the algorithm's behavior (not just its final output).
4. Save the visualization as a `.png` or `.gif` with a clear filename that matches the script name.
5. Add a section to `README.md` with: title, embedded image, and the key code snippet.
6. Update the table of contents in `README.md`.
7. Confirm the script runs end-to-end before finishing.

## When fixing or modifying an existing example

- Read the file fully first — understand what visualization it is trying to teach before changing anything.
- Keep the teaching intent intact. If your fix would obscure the algorithm, find another fix.
- If the visualization output changes, regenerate the saved image/gif and verify the README still reflects reality.
- Do not "clean up" unrelated code in the same file.

## Summary

> The reader is a human trying to build intuition about an ML/DL/RL algorithm as fast as possible by *seeing* it work. Every line you write should make that faster or clearer. If it doesn't, don't write it.
