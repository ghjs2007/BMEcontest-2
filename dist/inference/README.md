# 独立 event-stack 推理包

本目录由当前 promoted canonical release 自动生成，请勿手工修改 `event_stack/`；
重建命令：`python scripts/build_inference_distribution.py`。

安装精确记录的依赖后，即可对官方原始 `collect_data*.txt` 文件或其所在目录做预测：

```bash
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

支持 `--device cpu`。当前发布没有经过审计的 CUDA 适配器，强制 `--device gpu` 或
`--device cuda` 会被明确拒绝，而不是静默回退到 CPU。输出 JSON 遵循
`dist/schema/prediction.schema.json`。
