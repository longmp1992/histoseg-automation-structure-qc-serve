# histoseg_csr_qc — 基于 CSR 的 HistoSeg 结构质控与分割建议

输入一个 HistoSeg 输出 zip（或解压后的文件夹），自动给出：

1. 每个结构是否分割成功（PASS / PARTIAL / FAIL）
2. 分配给每个结构的 cluster 中是否存在高度聚集的亚区（HOTSPOT）
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
| `rl`（随机标记） | 仅从分配给该结构的 cluster 细胞中无放回抽 n 个 | 排除已分配结构自身密度不均后，该 cluster 是否仍聚集 |

**偏离指数** DI = mean over r∈[20, 100] µm of L_obs(r) / L_null(r) − 1
（DI > 0 聚集，≈ 0 随机，< 0 均匀）。r < 20 µm 受细胞体积硬核效应影响，不计入。

- p 值：用留一法得到模拟的 DI 分布，做单侧秩检验。
- Clark–Evans R = NND_obs / NND_null，作为辅助指标。
- 点数超过 20,000 时做独立稀疏化，L 函数对独立稀疏化保持不变。

判定以效应量 DI 为主，p 值只作门槛。细胞数上万时，极小的偏离也会显著，只看 p 值没有意义。

### 步骤 2 — 结构判定
| 指标 | 点集 | 窗口 | 零模型 |
|---|---|---|---|
| DI_tissue | 结构轮廓内、且 cluster 被分配给该结构的全部细胞 | 整个组织 | csr |
| DI_in | 结构轮廓内、且 cluster 被分配给该结构的全部细胞 | 自身结构 | csr |

- **FAIL**：DI_tissue 不显著（结构细胞在组织中并不集中）
- **PASS**：DI_in ≤ 0.10（轮廓内接近随机，轮廓解释了聚集）
- **PARTIAL**：0.10 < DI_in ≤ 0.30
- **FAIL**：DI_in > 0.30（轮廓内仍强烈聚集）

**单 cluster 硬规则：** 如果一个结构只分配了 1 个 cluster，则该结构直接判定为 `PASS`，
并设为 `split = NOT_NEEDED`；DI 指标仍输出供参考，但不用于推翻该判定。

EF = 1 − DI_in / DI_tissue 只作参考，不参与判定：面积占比大的结构 DI_tissue 天然偏小，会使 EF 偏低。

### 步骤 3 — 结构内 cluster
只对 StructureMap 分配给结构 s 的 cluster（在该结构轮廓内 ≥ 100 个细胞且占已分配细胞 ≥ 1%）计算 DI_csr 与 DI_rl。
落入该轮廓但属于其他结构或未分配的 cluster，不进入点集、随机标记背景、比例分母或图表。

- **HOTSPOT**：DI_rl ≥ 0.30、DI_csr ≥ 0.30，且 p_rl ≤ 0.05
- **CLUSTERED**：DI_rl ≥ 0.15，且 p_rl ≤ 0.05
- **DISPERSED** / **RANDOM-LIKE**：其余情况

HOTSPOT 均来自本结构已分配 cluster，并标记为 `SUBDOMAIN`；其他结构的 cluster 不再参与本结构 QC。

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
| `qc_cluster_DI_before_after.csv` | 每个已分配 cluster：原始 DI（全部细胞、整个组织）与最终 DI（所属结构轮廓内）；`qc_overview.png` 底部为对应的分组柱状图 |

## 默认阈值（均可调）
`n_sim=49, r=[20,100] µm, t_homog_pass=0.10, t_homog_partial=0.30, t_hotspot=0.30, t_mild=0.15,
t_edge=0.60, min_group_share=0.05, min_cells=100, min_share=0.01, max_points=20000`

这些阈值是根据两套数据（GSM9902814 0916、GSM9945415）定的经验值，还没有用人工标注的“好/坏结构”校准。

---

# 自动 ΔDI 结构分割（Serve 步骤 3，`histoseg.spatial_pathologist.auto_split`）

用户选择 ΔDI 阈值后，沿原始 StructureMap 树状图自动划分结构。DI 引擎与上文完全相同
（Ripley's L 与同一窗口内的 CSR 模拟比较，DI = 20–100 µm 上 L_obs/L_CSR 的平均 − 1），
划分由 Serve 应用自身的 HistoSeg 分区函数产生，参数与步骤 2 的高级参数一致。

1. **基线：** 每个 cluster 在整个组织中的 DI。
2. **全树探索：** 从“所有 cluster 为一个结构”开始，每次展开当前最高的分支点 v，用当前所有分支各作一个结构重跑 HistoSeg。
   - ΔDI(v) = Σ_{v 下的 cluster} [DI(v 的轮廓) − DI(所属子分支的轮廓)]。
   - DI 只用该 cluster 落在对应轮廓内的细胞计算。
   - “v 的轮廓”取 v 被切分前一刻的划分重新测量：HistoSeg 划分是竞争性的，其他分支被切分时 v 的轮廓也会移动。
   - 所有分支点都会被评估。
3. **按阈值 t 决定（从根开始）：**
   - ΔDI(v) ≥ t → 切开，继续处理子分支；
   - ΔDI(v) < t，且下面没有达标分支点 → 保留为一个结构；
   - ΔDI(v) < t，但下面有分支点 w 达标 → 按用户选择：
     - **提取贡献最大的子支（默认）：** v 不切，只把 w 的子分支中 ΔDI 贡献（各 cluster ΔDI 之和）最大的那一支单独成结构；
     - **连带切开父分支：** v 也切开并继续递归；
     - **停止：** v 保留，忽略 w。
3b. **或按结构数目切分（用户选择 K 个结构）：**
   - 所有分支点按 ΔDI 从大到小排名；
   - 依次选取排名靠前的分支点；选中一个分支点时，连带选中它尚未被选中的祖先分支点（否则它不是合法的结构边界）；
   - 若某分支点连同祖先会超过 K − 1 个切分名额，则跳过它，看下一个；选满 K − 1 个为止（K 不超过 cluster 数）；
   - 报告中标明每个被切开的分支点是“按 ΔDI 排名选中”还是“作为某分支点的祖先被连带切开”。
4. **最终划分：** 用得到的结构重跑 HistoSeg，并计算每个 cluster 在最终轮廓中的 DI。
5. **重新切分：** 全树探索的结果缓存在 `autosplit_exploration.pkl`；在界面上改变阈值、规则或结构数后点 *Re-split*，
   只重跑第 3–4 步（一次划分 + 各 cluster 的 DI），不再重新探索整棵树。

## 输出（`autosplit_*`）
| 文件 | 内容 |
|---|---|
| `autosplit_report.md` | 报告：最终结构、每个分支点 ΔDI 与判定、提取的分支、每个 cluster 分割前（整个组织）/后（最终结构）的 DI、每个分支点切分时各 cluster 的前后 DI |
| `autosplit_branch_points.csv` | 每个分支点：高度、子分支、Σ DI 前/后、ΔDI、判定、原因 |
| `autosplit_branch_point_cluster_DI.csv` | 每个分支点切分时每个 cluster 的 DI 前/后与 ΔDI |
| `autosplit_cluster_DI_before_after.csv` | 每个 cluster：整个组织 DI、最终结构 DI、ΔDI、落在自身轮廓内的比例 |
| `autosplit_structures.csv`, `autosplit_histoseg_structures.txt` | 最终结构（每行一个，可直接用于步骤 2） |
| `autosplit_extracted_branches.csv` | 被提取的子分支及各子支贡献 |
| `autosplit_dendrogram.png`, `autosplit_partition.png` | 标注 ΔDI 与判定的树状图；最终结构的空间划分 |
| `autosplit_cluster_DI_before_after.png` | 每个 cluster 原始（整个组织）与最终（所属结构内）DI 的分组柱状图 |

## 运行时间
- **工作量：** 每个分支点需要 1 次 HistoSeg 划分，外加该分支点下 cluster 数 × 2 次 DI 检验。
- **并行数：** 默认 `min(2, CPU)` 个并行进程，可用环境变量 `HISTOSEG_QC_WORKERS` 调整。
- **加速：** 大样本可降低 Monte Carlo 次数（界面滑块，最小 19）。
