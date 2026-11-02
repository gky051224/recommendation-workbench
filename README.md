# Recommendation Workbench

一个独立实现的、零第三方运行依赖的推荐算法实验工具。支持可解释的 UserCF / ItemCF、冷启动回退、类型筛选和可复现的离线评估。

## 快速开始

需要 Python 3.11 或更新版本。在仓库根目录执行，Windows 和 Linux 命令相同：

```console
python -m reclab demo local-data
python -m reclab evaluate local-data/ratings.csv --report local-reports/evaluation.json
python -m reclab recommend local-data/ratings.csv --user viewer-000 --mode item --k 5 --top 5 --catalog local-data/items.csv
python -m unittest discover -s tests -v
```

无需安装 numpy / pandas。也可以在虚拟环境中用 `python -m pip install -e .` 安装，再用 `reclab` 命令。`demo` 遇到已有同名数据文件会拒绝覆盖。

## 功能与方法

- **更可靠的实验流程**：按用户划分训练、验证、测试集，目标比例为 70% / 15% / 15%。仅使用验证集选择 K，再用训练 + 验证集重训，最后评估测试集。报告包含 RMSE、MAE、实际样本量、各集合 SHA-256 和固定种子。
- **有基线的比较**：迭代拟合带正则化的用户偏差和物品偏差，作为独立基线；协同过滤在该基线的残差上计算余弦相似度，并按共同评分数收缩相似度。仅采用正相似度近邻。
- **可解释的推荐**：输出预测分、未截断的基线、邻居 ID、相似度及评分残差。残差加权平均加到基线上，最后截断到 1～5 分。无可用邻居时回退到基线，报告 `fallback: true`。
- **冷启动和稳定排序**：未知用户可得到物品偏差推荐；未知物品退回用户偏差；两者未知时使用全局均分。已评分物品被排除，分数相同时按字符串 ID 排序。
- **输入校验**：拒绝重复用户／物品评分对、空 ID、非有限评分和超出 1～5 范围的评分，避免重复评分跨集合泄漏。不要求 ID 连续或为数字。

```console
python -m reclab evaluate local-data/ratings.csv --seed 7 --k 3 5 10 20 --report local-reports/seed7.json
python -m reclab recommend local-data/ratings.csv --user new-viewer --mode baseline --top 5
python -m reclab recommend local-data/ratings.csv --user viewer-000 --catalog local-data/items.csv --genre mystery --top 5
```

`recommend` 使用传入的完整数据训练，适合最终推荐；`evaluate` 自己切分输入，适合离线比较。要在推荐时使用评估选出的 K，请把报告中的 `best_k` 传给 `--k`。

## 自己的数据

评分 CSV 必须包含以下列（可有额外列；UTF-8 或带 BOM 的 UTF-8 均可）：

```csv
user_id,item_id,rating
alice,film-a,4.5
bob,film-a,3
```

可选目录 CSV 格式：`item_id,title,genre`。`--genre` 需要同时提供 `--catalog`，按 genre 字段完整匹配。目录项可以没有历史评分，此时会触发基线回退。每个目录 ID 必须唯一。

少于 3 条评分的用户全部保留在训练集；其他用户至少保留 1 条训练评分、1 条验证评分、1 条测试评分。因此小样本数据的实际比例会偏离目标比例。没有任何用户达到 3 条评分时，评估返回明确错误。

## 结构

```text
reclab/data.py          CSV 校验、合成数据、按用户切分
reclab/model.py         正则化基线、协同过滤、推荐解释
reclab/evaluation.py    验证集调参、最终测试、数据指纹
reclab/__main__.py      demo / evaluate / recommend 命令
tests/                 算法边界、数据隔离、命令行集成测试
.github/workflows/     Windows / Linux 自动测试
docs/                  背景和本地验证记录
examples/              合成数据上的实际评估报告
```

## 限制与背景

这是面向教学的小数据工具，使用内存字典和按需缓存，适合数千条评分的实验；没有大规模稀疏矩阵优化或在线服务。当前评估是随机按用户留出，**不代表时间顺序预测**。合成数据只用于验证流程，误差不能解释为真实电影推荐质量。`fallback_rate` 指无正相似度邻居的预测比例，基线模型的该值恒为 1，不是排名覆盖率。

项目需求源于对 [wang01-12/recommendation-algorithm-lab](https://github.com/wang01-12/recommendation-algorithm-lab) 的审查；本仓库独立实现，没有复制其源代码、评分数据、电影海报、字体或提交历史，也不声称拥有原项目。详细审查和权属边界见 [背景说明](docs/PROVENANCE.md)。原项目权利仍属于其各自权利人。本项目尚未指定开源许可证。
