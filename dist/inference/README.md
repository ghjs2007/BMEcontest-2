# Standalone Event-Stack Inference

This directory is generated from the promoted canonical release. Do not edit
`event_stack/` manually; rebuild it with `python scripts/build_inference_distribution.py`.

Install the exact recorded dependencies, then predict an official raw
`collect_data*.txt` file or a directory containing such files:

```bash
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

`--device cpu` is supported. The promoted release has no audited CUDA adapter,
so forcing `--device gpu` or `--device cuda` is rejected rather than silently
falling back to CPU. The JSON result follows `dist/schema/prediction.schema.json`.
