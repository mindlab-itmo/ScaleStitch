# ScaleStitch

Inference and evaluation code for **When Size Helps and Hurts: Physical Measurements in 3D Question Answering**.

The pipeline assembles cached object, image, and directed-relation features, runs a frozen 3DGraphLLM under measurement controls, scores answers, and computes paired statistical comparisons.

Supply prepared features, physical measurements, question annotations, and pretrained model checkpoints. ULIP-2 features also require a fitted bridge checkpoint. Feature extraction and bridge fitting precede this pipeline.

From the repository root in PowerShell or a Linux shell:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/scale_interface/tests
```

See the [run commands](experiments/scale_interface/README.md#run) for preparation, inference, scoring, and analysis, and the [experimental protocol](experiments/scale_interface/PROTOCOL.md) for input formats and controls. The implementation and editable configuration examples are in [experiments/scale_interface](experiments/scale_interface/).
