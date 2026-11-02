# 实际验证记录

日期：2026-11-02。环境：Windows、Python 3.13.12。运行不需要第三方库。

## 自动化测试

```console
python -m unittest discover -s tests -v
```

结果：31 项测试全部通过。在新建的独立虚拟环境中也全部通过；`pip install -e .` 已成功构建并安装命令行入口。覆盖 CSV / ID / 分值校验、重复评分拒绝、留出集互斥、稀疏用户、冷启动、无共同评分、零残差、已看物品排除、排序、推荐理由重算、指标手算、K 参数检查、测试标签不参与拟合及选择、命令行完整流程和错误退出码。

特别检查了：只修改测试评分不会改变验证曲线及所选 K；所有拟合只收到训练集或训练 + 验证集，没有测试集记录。

## 合成数据完整实验

```console
python -m reclab demo examples/synthetic
python -m reclab evaluate examples/synthetic/ratings.csv --report examples/evaluation.json
python -m reclab recommend examples/synthetic/ratings.csv --user viewer-000 --mode item --k 3 --top 5 --catalog examples/synthetic/items.csv
```

仓库已经包含生成结果，重跑 `demo` 时请换一个空目录。相同种子 42 下，独立生成的数据文件逐字节相同；两次评估报告也逐字节相同。

虚构数据：80 位用户、36 部电影、1,440 条评分。实际切分：训练 1,120 条，验证 160 条，测试 160 条。每位用户仅有 18 条评分，比例取整后与目标 70/15/15 不完全一致。

| 模型 | 验证集所选 K | 测试 RMSE | 测试 MAE | 回退比例 |
| --- | ---: | ---: | ---: | ---: |
| 正则化偏差基线 | 不适用 | 0.917666 | 0.774958 | 100% |
| UserCF | 5 | 0.373920 | 0.301749 | 0% |
| ItemCF | 3 | 0.431858 | 0.341024 | 0% |

完整曲线、参数和集合指纹见 [evaluation.json](../examples/evaluation.json)。推荐结果见 [recommendations.json](../examples/recommendations.json)。合成数据带有明确的人为偏好结构，以上误差不支持对真实用户、真实电影或生产效果的判断。

## 持续集成

工作流会在 Windows / Linux、Python 3.11 / 3.13 上运行测试和完整合成数据评估。[首次远端运行](https://github.com/gky051224/recommendation-workbench/actions/runs/41398809641) 的四种环境均已通过。随后更新了检出代码和安装 Python 的官方 Actions 版本，以消除旧 Node.js 运行时的弃用提示；最新提交的状态以 GitHub Actions 页面为准。
