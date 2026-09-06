# 无 Agent 版本的基准开发

## 第一阶段：固定基线（已完成）

基线为 `benchmarks/v1/baseline/`，来自原 `onset_refined` 正式输出。工作参考独立冻结在 `benchmarks/v1/reference/`，清单记录来源提交与 SHA256。修改参考或基线必须创建新版本，不覆写 v1。评估器校验文件哈希，不读取可变的 current 指针，也不参与推理。

默认运行 `python run_workbench_pipeline.py episode_000000` 使用原流程。实验复核只有明确添加 `--experimental-review` 才运行。工作台默认隐藏旧实验报告；通过 annotation API 的 `experimental_review=true` 可读取历史报告。

| 轨迹 | 子任务数 | 技能及手臂序列与参考 | 帧标签与参考一致率 |
| --- | --- | --- | --- |
| episode_000000 | 11 | 一致 | 98.6577% |
| episode_000001 | 7 | 一致 | 99.4924% |

这些是对工作参考的一致性指标，不是精确物理接触准确率，也不涵盖动作文本、目标属性或未来轨迹泛化。16 个内部边界中 9 个一致，7 个偏差 1—3 帧。具体列于 `benchmarks/v1/cases.json`，目前没有已确认的算法错例，不据此调整阈值。

## 可复现对照

```text
python benchmark.py --candidate-dir outputs/automatic/onset_refined/annotations --output outputs/benchmark/current.json
python -m pytest -q test_benchmark.py test_workbench.py test_multiview_review.py test_fact_review.py test_static_display.py
```

候选必须覆盖两条完整轨迹。报告包括结构检查、技能/手臂序列编辑距离、帧标签一致率、逐边界相对参考的前后差异、相对基线变动帧数及文件哈希。序列不一致时禁止按位置强行匹配边界。整体 JSON 变化另行标记，不能只看帧数掩盖文本变化。

本次两条默认流程缓存复跑成功；结果与冻结基线完整内容一致，45 项检查通过。缓存运行不代表重新推理的稳定性或真实在线耗时，未以它估算在线成本。

## 后续每轮实验的必要记录

1. 具体错例 ID、用户已确认的预期、可复查的原始帧与视角。模型分歧不是错误真值。
2. 一个待验证假设、一个主要改动；先记录再运行，不按参考帧反调阈值。
3. 两条完整轨迹的同一份对照报告，列出改善、退化和未变项；人工核查文本及目标属性。
4. 模型、参数、请求及图像、缓存命中、实际 API 次数、用量与耗时。区分冷调用和缓存运行。
5. 目标错例确有改善且已有正确项不退化后才考虑采用；不能仅因更接近不确定参考就通过。评估器不自动批准改动。

## 当前下一步

先核查 7 项差异中证据较明确的运动/静止转换，再讨论遮挡下按压边界。不直接将差异改成参考值。若现有数据不能证明谁对，只提交具体片段请用户裁定；未裁定前保持基线，不扩大约束或引入 Agent。
