# Credits

This recipe combines and records work; it does not claim authorship of the base
engine, model, quantizer, original four-node family or all merged improvements.

- Ash Hart and TensorFold contributors: native Zig runtime, server, CUDA and
  model infrastructure. Apache-2.0, with inherited third-party notices.
- Albert Lee / BertholomusAI: DeepSeek-V4.1 native family, four-node 2D split,
  EXL3 checkpoint, kernel kit and original recipe. See `engine/ATTRIBUTION.md`.
- MiaAI-Lab and the individually credited patch authors: the upstream patch
  collection merged before this optimization campaign. Original notices remain
  in the reconstructed source and `engine/` license files.
- DeepSeek: model architecture, checkpoint and original reference material.
- Turboderp and ExLlamaV3 contributors: EXL3 format and conversion/kernel work.
- Anton Starikov and collaborators: serving integration, local source changes,
  controlled qualification, transport comparison and this recipe.

The source overlay contains inherited engine upgrades as well as local changes;
`engine/changed-files.json` is a reconstruction delta, not an authorship list.
No NInfer/Qwen experiment is presented as an upstream ExLlama defect or as a
measured improvement to this DeepSeek profile.
