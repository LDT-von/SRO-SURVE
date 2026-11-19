# 在 3090 上运行 SRO-Surv

当前默认协议是 BLCA DSS、`5fold_uni2h` fold 0、seed 3、每病例最多采样 4096 个 patch、batch size 8、35 个训练 epoch。沿用项目已有的 PyTorch 环境，在项目根目录打开 PowerShell。

## 核对输入

| 输入 | 3090 上要找的文件 |
| --- | --- |
| 临床 | `05_clinical` 中的 BLCA 表，需要 `case id`、`survival_months_dss`、`censorship_dss` 和 `wsi` 列 |
| RNA | `04_omics_rna` 中的 `blca_rna_inter.csv`，行为基因，列为患者 |
| 通路签名 | 原 SlotSPE 的 `combine_signatures.csv`，与 BLCA RNA 取交集后应保留 329 条通路 |
| 分折 | `06_splits` 中的 `5fold_uni2h\blca\fold_0.csv` |
| 病理 | `03_slide_feat_uni2h` 中 BLCA 的逐片 UNI2-h `.h5` 特征。若只有 tar，只解包 BLCA 所需的归档。若已有确认同源的 UNI2-h `.pt`，也可直接使用其所在目录。 |

将下列示例替换为实际文件路径。`--features-root` 指向直接包含逐片文件的那一级目录。训练使用预提取特征，无需准备 `01_raw_wsi`。

```powershell
$sroInputs = @(
    '--clinical-csv', 'E:\TCGA\05_clinical\all\blca.csv',
    '--rna-csv', 'E:\TCGA\04_omics_rna\raw_rna_data_inter\blca_rna_inter.csv',
    '--signature-csv', 'E:\TCGA\04_omics_rna\signatures\combine_signatures.csv',
    '--split-csv', 'E:\TCGA\06_splits\5fold_uni2h\blca\fold_0.csv',
    '--features-root', 'E:\TCGA\03_slide_feat_uni2h\TCGA_WSI_BLCA'
)
python -X utf8 -B scripts/check_sro_surv.py --device cuda:0
python -X utf8 -B scripts/prepare_sro_surv.py @sroInputs
```

第一条用小规模人工数据检查 CUDA 和训练流程。第二条只检查 CSV 表头及特征目录，患者匹配在准备阶段进行。

## 生成缓存

准备阶段会把四份 CSV 复制到 `local_evidence` 下，记录原始路径和 SHA-256，并按患者写出全部真实 patch 的 float32 缓存。切片特征直接从原路径读取。确认仓库所在盘能容纳这份缓存，并在后续运行中保留准备目录的绝对路径。

```powershell
python -X utf8 -B scripts/prepare_sro_surv.py @sroInputs --prepare --output-root local_evidence/blca_f0_prepared
```

核对输出中的训练、验证和排除人数及 329 条通路。准备目录下的 `prepared/manifest.json` 记录分折与排除信息。

## 短跑对照槽冗余

按上面的默认目录准备好缓存后，在项目根目录运行一条命令。

```powershell
python -X utf8 -B scripts/compare_sro_surv.py
```

默认每组跑 5 轮。两组共用 UNI2-h 特征、分折、初始种子和 patch 抽样规则。SRO 组在病理侧 10 次槽读取都应用正则，权重在第 5 轮升到 `0.10`。`--epochs 1` 可用于快速核对流程。要调整显存开销，可给两组同时指定 `--batch-size 4`。重复运行时用 `--output-root` 指定 `local_evidence/` 下的新目录。

终端表格对比同一末轮。报告位于 `local_evidence/sro_redundancy_compare/redundancy_comparison.json`，其中还记录每个病理槽移除后的 NLL 变化。

| 指标 | 怎么看 |
| --- | --- |
| `pathology_read_BC_round_10` | 最后一轮槽读取重叠，越接近 1 表示槽越倾向读取相同 patch。也可查看第 1、5 轮的变化。 |
| `pathology_slot_pair_cosine` | 最终病理槽向量的两两余弦相似度。读入重叠与槽余弦一起下降，支持病理槽冗余缓解。 |
| `pathology_across_patient_std_rms` | 槽在不同患者之间的变化幅度。接近零时要留意槽是否学成了近似常量。 |
| `validation_C_index` 和 `validation_NLL` | 分别看风险排序和生存损失，结合几何指标判断变化是否伴随验证表现下降。 |

这次短跑用于观察单折开发验证中的早期变化。判断效果能否持续，需要更长预算和跨折复现。

## 单独跑默认完整预算

```powershell
python -X utf8 -B scripts/run_sro_surv.py --run --prepared-root local_evidence/blca_f0_prepared --output-root local_evidence/sro_blca_f0_s3
```

这条命令只跑 SRO 组，默认 35 轮。首次运行请指定一个新目录。
