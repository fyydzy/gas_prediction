# gas-prediction

省级天然气销量**旬度**预测项目。从原始日度销量与气温数据出发，聚合为旬度样本，经统一特征工程后，对比多种机器学习、深度学习与时序基础模型在固定验证窗口上的表现。

当前主流程在 `scripts1/` 与 `src/gas_prediction/feature_engineering1.py`；`scripts/` 为较早的月度/实验脚本，可作参考。

## 环境要求

- Python `>=3.11, <3.14`（见 `.python-version`）
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖

```bash
# 安装依赖
uv sync

# 将源码包加入 Python 路径（Windows PowerShell）
$env:PYTHONPATH="src"

# Windows CMD
set PYTHONPATH=src
```

## 项目结构

```
gas_prediction/
├── data/                          # 数据目录（默认被 .gitignore 忽略）
│   ├── original_data1/
│   │   ├── 销量/{省份}/           # 各省日度销量 Excel
│   │   └── 温度/                    # 气温原始表
│   └── processed_data1/
│       └── {省份}.xlsx              # 旬度加工结果（建模输入）
├── src/gas_prediction/
│   ├── feature_engineering1.py    # 旬度特征工程流水线
│   ├── forecast_common1.py        # 数据加载、旬度切分、指标
│   ├── feature_engineering.py     # 旧版月度特征工程
│   └── forecast_common.py           # 旧版公共工具
├── scripts1/                      # 旬度建模主脚本
├── scripts/                       # 旧版/实验脚本
└── output1/                       # 模型输出（默认被 .gitignore 忽略）
```

## 数据准备

### 1. 原始数据

将数据放入：

- `data/original_data1/销量/{省份}/` — 各省日度天然气销量 Excel
- `data/original_data1/温度/` — 气温数据（脚本按省会/代表城市匹配，见 `process_sales_tenday.py` 中 `PROVINCE_TO_TEMP_REGION`）

### 2. 生成旬度 processed 表

```bash
uv run scripts1/process_sales_tenday.py
```

按省输出到 `data/processed_data1/{省份}.xlsx`，每行一个旬，核心列包括：


| 列名                                   | 说明                  |
| ------------------------------------ | ------------------- |
| `date`                               | 旬起始日（1 / 11 / 21 日） |
| `gas_sales`                          | 旬内天然气销量             |
| `avg_temp` / `max_temp` / `min_temp` | 旬内气温统计              |
| `HDD`                                | 采暖度日数（基温 18°C，旬内累加） |
| `extreme_cold_days`                  | 旬内极端低温天数            |


### 3. 切换建模省份

在 `src/gas_prediction/forecast_common1.py` 中修改：

```python
PROCESSED_PROVINCE = "河北"
```

## 特征工程

`build_features_pipeline()`（`feature_engineering1.py`）在 processed 表基础上构造建模特征：

1. **气象衍生**：`temp_range`
2. **时间特征**：`time_index`、`month_sin/cos`、`tenday_in_month`、`is_heating_season`、`spring_rework_peak`
3. **滞后特征**：`Lag_36`（约一年前同一旬）
4. **交互/非线性**：`HDD_squared`、`HDD_cross_Lag_36`、`HDD_cross_HeatingSeason`、`ColdDays_cross_Lag_36`

流水线会因 `Lag_36` 自动丢弃头部 36 行含 NaN 的样本。

导出指定时段特征示例（2021–2026 年每年 3 月上旬）：

```bash
uv run scripts1/extract_march_processed.py
uv run scripts1/extract_march_processed.py --province 河北 --output output1/custom.xlsx
```

## 训练与评估约定

多数 `scripts1/run_*.py` 采用统一的旬度切分：


| 划分  | 日期范围                        | 说明       |
| --- | --------------------------- | -------- |
| 训练集 | `date <= 2025-06-21`        | 截至该旬     |
| 测试集 | `2025-11-01` ~ `2026-03-21` | 闭区间，旬起始日 |


默认省份为河北，结果写入 `output1/`。

## 模型脚本

在 `PYTHONPATH=src` 前提下运行：

### 表格特征 + 监督学习


| 脚本                     | 模型                        | 输出示例                           |
| ---------------------- | ------------------------- | ------------------------------ |
| `run_lightgbm.py`      | LightGBM + 网格搜索           | `output1/lightgbm/`            |
| `run_xgboost.py`       | XGBoost                   | `output1/xgboost/`             |
| `run_catboost.py`      | CatBoost                  | `output1/catboost/`            |
| `run_random_forest.py` | 随机森林                      | `output1/random_forest/`       |
| `run_ridge.py`         | Ridge                     | `output1/ridge/`               |
| `run_lasso.py`         | Lasso                     | `output1/lasso/`               |
| `run_ElasticNet.py`    | ElasticNet                | `output1/elasticnet/`          |
| `run_lstm.py`          | LSTM（连续特征 + 月份 Embedding） | `output1/{省}_lstm_test_*.xlsx` |


```bash
uv run scripts1/run_lightgbm.py
uv run scripts1/run_catboost.py
```

树模型脚本通常输出：测试集预测 Excel、验证指标、特征重要性图、网格搜索结果 CSV。

### 时序 / 基础模型


| 脚本                 | 说明                      |
| ------------------ | ----------------------- |
| `run_prophet_0.py` | Prophet + XGBoost 残差修正  |
| `run_prophet_1.py` | Prophet + LightGBM 残差修正 |
| `run_sarimax.py`   | SARIMAX 多步预测            |
| `run_chronus.py`   | Chronos 零样本/上下文预测       |
| `run_timesfm.py`   | TimesFM 验证              |


两阶段 Prophet 类脚本使用 `forecast_common1` 中的 `AS_OF_DATE`、桥接旬与验证旬逻辑，输出文件名含 `ctx` 与预测步数。

### 模型对比

汇总各模型测试集预测，生成对比图与指标分析：

```bash
uv run scripts1/compare_models.py
```

默认输出到 `output1/model_comparison/`。

## LSTM 中的 Embedding

`run_lstm.py` 将月份作为类别特征：

- 从 `date` 提取 `month_idx`（1–12）
- `nn.Embedding(13, 4)` 映射为 4 维向量
- 与连续特征（`Lag_36`、`HDD`、`is_heating_season`）在最后一维拼接后送入 LSTM
- 序列长度 `SEQ_LENGTH=9`（约 3 个月 × 3 旬）

## 修改实验配置

常见可调项（各脚本文件顶部常量）：

- `TRAIN_END` / `TEST_START` / `TEST_END` — 训练/测试时间窗
- `MODEL_FEATURE_COLS` — 参与建模的特征子集
- 网格搜索超参、`RANDOM_SEARCH_N_TRIALS` 等

全局默认值见 `forecast_common1.py`：

```python
AS_OF_DATE = "2025-06-21"
VAL_START = "2025-11-01"
VAL_END = "2026-03-21"
PROCESSED_PROVINCE = "河北"
```

## 输出说明

`output1/` 下典型产物：

- `{省}_{模型}_test_{开始}_to_{结束}.xlsx` — `test_forecast`（预测 vs 真实）、`metrics` 工作表
- `{模型}/` — 特征重要性图、网格搜索 CSV
- `model_comparison/` — 多模型对比图与文字分析

## 依赖概览

主要 Python 包（完整列表见 `pyproject.toml`）：

- 数据处理：`pandas`、`numpy`、`openpyxl`
- 机器学习：`scikit-learn`、`lightgbm`、`xgboost`、`catboost`
- 时序：`prophet`、`statsmodels`、`pmdarima`、`neuralforecast`、`timesfm`
- 深度学习：`torch`
- 可视化：`matplotlib`

