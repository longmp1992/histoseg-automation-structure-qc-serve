# histoseg_csr_qc — 基于 CSR 的 HistoSeg 结构质控与分割建议

输入一个 HistoSeg 输出 zip（或解压后的文件夹），自动给出：

1. 每个结构是否分割成功（PASS / PARTIAL / FAIL）
2. 每个结构内部是否存在高度聚集的 cluster（HOTSPOT），以及它们的角色
3. 是否建议进一步分割，并按原始 StructureMap 树状图给出可直接回填 HistoSeg 的新 `cluster_ids` 分组

```bash
python histoseg_csr_qc.py "histoseg_outputs - 2026-09-16T110313.955.zip" --structuremap row_coph.csv -o qc_out
```

所需文件：`cells_with_structure_partition.parquet`、`structure_contour_metrics.json`，
以及同一次运行的 StructureMap 矩阵（放进 zip，或用 `--structuremap` 指定）。
所有阈值均可通过命令行覆盖，例如 `--t-hotspot 0.4 --n-sim 99`。

---

## 算法

### 步骤 0 — 观察窗口
用 `histoseg.contour.multi_structure._build_structure_isolines` 按 json 中的参数重建分区栅格，
并用细胞的 `isoline_structure_id` 校验（要求一致率 ≥ 99.5%）。
导出的 contour 在图像边缘不闭合，不能可靠地还原成多边形，因此不用它们。
若 histoseg 不可用或校验失败，退回“细胞多数标签栅格 + 最近邻填充”，报告中会注明窗口来源和一致率。

### 步骤 1 — 统一检验引擎
对任意点集 X 与窗口 W，在**同一窗口**内与 Monte Carlo 零模型比较（边界效应相互抵消）：

| 零模型 | 模拟方式 | 回答的问题 |
|---|---|---|
| `csr` | 在 W 内均匀撒 n 个点 | 相对完全空间随机是否聚集 |
| `rl`（随机标记） | 从该结构全部细胞中无放回抽 n 个 | 排除结构自身密度不均后，该 cluster 是否仍聚集 |

**偏离指数** DI = mean over r∈[20, 100] µm of L_obs(r) / L_null(r) − 1
（DI > 0 聚集，≈ 0 随机，< 0 均匀）。r < 20 µm 受细胞体积硬核效应影响，不计入。

- p 值：用留一法得到模拟的 DI 分布，做单侧秩检验。
- Clark–Evans R = NND_obs / NND_null，作为辅助指标。
- 点数超过 20,000 时做独立稀疏化，L 函数对独立稀疏化保持不变。

判定以效应量 DI 为主，p 值只作门槛。细胞数上万时，极小的偏离也会显著，只看 p 值没有意义。

### 步骤 2 — 结构判定
| 指标 | 点集 | 窗口 | 零模型 |
|---|---|---|---|
| DI_tissue | 结构内全部细胞 | 整个组织 | csr |
| DI_in | 结构内全部细胞 | 自身结构 | csr |

- **FAIL**：DI_tissue 不显著（结构细胞在组织中并不集中）
- **PASS**：DI_in ≤ 0.10（轮廓内接近随机，轮廓解释了聚集）
- **PARTIAL**：0.10 < DI_in ≤ 0.30
- **FAIL**：DI_in > 0.30（轮廓内仍强烈聚集）

EF = 1 − DI_in / DI_tissue 只作参考，不参与判定：面积占比大的结构 DI_tissue 天然偏小，会使 EF 偏低。

### 步骤 3 — 结构内 cluster
对结构 s 内每个 cluster（≥ 100 个细胞且占结构 ≥ 1%），计算 DI_csr 与 DI_rl。

- **HOTSPOT**：DI_rl ≥ 0.30、DI_csr ≥ 0.30，且 p_rl ≤ 0.05
- **CLUSTERED**：DI_rl ≥ 0.15，且 p_rl ≤ 0.05
- **DISPERSED** / **RANDOM-LIKE**：其余情况

HOTSPOT 再按“是否属于本结构”和“距轮廓比”分角色。
距轮廓比 = 该 cluster 细胞到结构边界距离的中位数 ÷ 结构全部细胞的对应中位数。

| 角色 | 条件 | 含义 |
|---|---|---|
| SUBDOMAIN | 属于本结构 | 本结构内部的亚区，分割候选 |
| EMBEDDED_FOREIGN | 属于其他结构，距轮廓比 > 0.6 | 结构内部未被切出的外来岛，分割候选 |
| BOUNDARY_SPILLOVER | 属于其他结构，距轮廓比 ≤ 0.6 | 贴着轮廓的溢入细胞，属于轮廓精度问题，**不**作为分割依据 |

### 步骤 4 — 分割建议（CSR 决定“是否分”，原始 StructureMap 决定“怎么分”）
不自己计算任何 cluster 间距离或共定位，直接使用同一次 HistoSeg 运行的 **原始 StructureMap 矩阵**
（row_coph 共表型矩阵，cluster × cluster CSV）。

- **矩阵来源：** 输入 zip/文件夹中文件名含 `row_coph` / `cophenetic` / `structuremap` 的 CSV，或用 `--structuremap` 指定。
- **读取校验：** 行列 cluster 一致、矩阵对称、在该结构的 cluster 上满足超度量性（ultrametric）。
  共表型矩阵本身就唯一确定了树（合并高度即矩阵值），所以只是“读出”这棵树，不重新估计。
  若传入的是原始距离矩阵（不满足超度量性），会直接报错。
- **切树规则：** 取该结构 `cluster_ids` 在 StructureMap 上的子树，从子树根开始：
  - 分支内**同时有**本结构 HOTSPOT（SUBDOMAIN）和非 HOTSPOT cluster → 在该节点切开，递归处理各子分支（高度相同的连续合并视为一个多叉节点）；
  - 分支内**全部是** HOTSPOT，或**全部不是** → 停止。
  - 每个 HOTSPOT 分支成为一个建议新结构（`S.1`, `S.2`, …）；所有非聚集分支的 cluster **合并为一类** `S.rest`，不单独成结构。
- **建议等级：**
  - **RECOMMENDED**：结构非 PASS，且（子树能被切成 ≥ 2 支，或存在占比 ≥ 5% 的内嵌外来岛）
  - **OPTIONAL**：结构 PASS，子树能被切成 ≥ 2 支，且最大 SUBDOMAIN DI_rl ≥ 0.6
  - **REVIEW**：结构 FAIL，但无可切分的 HOTSPOT
  - **NOT_NEEDED**：其余情况
- **建议类型：** `subdomain`（按树切分，给出各分支的 `cluster_ids`）/ `carve_island`（内嵌外来岛不在本结构的树上，只作提示，可降低 `min_component_pixels`）。
- **小分支标记：** 占结构 < 5% 的 HOTSPOT 分支会被标注；工具不会自动合并。
- **没有矩阵时：** 仍给出判定与 HOTSPOT，但分割方案标注“需要原始 StructureMap 矩阵”。

## 输出
| 文件 | 内容 |
|---|---|
| `qc_report.md` | 中文报告：判定、聚集 cluster、分割建议 |
| `qc_overview.png` | 结构 DI 对比 + 各结构内 cluster 的 DI_rl |
| `qc_structures.csv` | 结构级指标与判定 |
| `qc_clusters.csv` | cluster 级指标、状态、角色 |
| `qc_split_suggestions.json` | 机器可读的建议：树切分分支、新 `cluster_ids`、外来岛、边界溢入 |
| `qc_tests.csv`, `qc_L_ratio_curves.csv` | 全部检验结果与 L 比值曲线 |
| `partition_labels.npz` | 所用窗口栅格 |

## 默认阈值（均可调）
`n_sim=49, r=[20,100] µm, t_homog_pass=0.10, t_homog_partial=0.30, t_hotspot=0.30, t_mild=0.15,
t_edge=0.60, min_group_share=0.05, min_cells=100, min_share=0.01, max_points=20000`

这些阈值是根据两套数据（GSM9902814 0916、GSM9945415）定的经验值，还没有用人工标注的“好/坏结构”校准。
