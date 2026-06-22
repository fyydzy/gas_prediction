# gas-prediction

省级天然气销量预测项目，**同时保留月度与旬度两套完整流程**。两套代码相互独立：数据目录、特征工程、公共配置、运行脚本、输出目录均不同，请勿混用。


|        | **旬度（当前）**                                   | **月度（早期版本，保留对比）**                           |
| ------ | -------------------------------------------- | ------------------------------------------- |
| 时间粒度   | 上/中/下旬（`date` 为 1 / 11 / 21 日）               | 自然月（`month` 为 `YYYY-MM`）                    |
| 数据加工脚本 | `scripts1/process_sales_tenday.py`           | `scripts/process_all.py`                    |
| 原始数据目录 | `data/original_data1/`                       | `data/original_data/`                       |
| 建模输入目录 | `data/processed_data1/`                      | `data/processed_data/`                      |
| 特征工程   | `src/gas_prediction/feature_engineering1.py` | `src/gas_prediction/feature_engineering.py` |
| 公共工具   | `src/gas_prediction/forecast_common1.py`     | `src/gas_prediction/forecast_common.py`     |
| 建模脚本目录 | `scripts1/`                                  | `scripts/`                                  |
| 模型输出目录 | `output1/`                                   | `output/`                                   |
| 滞后特征   | `Lag_36`（约 12 个月 × 3 旬）                      | `Lag_12`（12 个月）                             |
| 训练截止   | `2025-06-21`（旬起始日）                           | `2025-06`                                   |
| 测试区间   | `2025-11-01` ~ `2026-03-21`                  | `2025-11` ~ `2026-03`                       |


## 推荐使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理环境

### 安装命令

在项目根目录执行：

```bash
uv sync
```

---

## 项目结构

```
gas_prediction/
├── data/
│   ├── original_data/          # 【月度】原始销量 + 温度
│   ├── processed_data/         # 【月度】加工后的建模表
│   ├── original_data1/         # 【旬度】原始销量 + 温度
│   │   ├── 销量/{省份}/
│   │   └── 温度/
│   └── processed_data1/        # 【旬度】加工后的建模表
├── src/gas_prediction/
│   ├── feature_engineering.py    # 【月度】
│   ├── forecast_common.py        # 【月度】
│   ├── feature_engineering1.py   # 【旬度】
│   └── forecast_common1.py       # 【旬度】
├── scripts/                      # 【月度】建模与实验脚本
├── scripts1/                     # 【旬度】建模与实验脚本
├── output/                       # 【月度】输出
└── output1/                      # 【旬度】输出
```

---

## 数据准备

### 旬度

原始数据放入 `data/original_data1/销量/{省份}/` 与 `data/original_data1/温度/`，然后：

```bash
uv run scripts1/process_sales_tenday.py
```

输出 `data/processed_data1/{省份}.xlsx`。核心列：


| 列名                                   | 说明                  |
| ------------------------------------ | ------------------- |
| `date`                               | 旬起始日（1 / 11 / 21 日） |
| `gas_sales`                          | 旬内销量                |
| `avg_temp` / `max_temp` / `min_temp` | 旬内气温                |
| `HDD`                                | 采暖度日（基温 18°C，旬内累加）  |
| `extreme_cold_days`                  | 旬内极端低温天数            |


切换省份：修改 `forecast_common1.py` 中 `PROCESSED_PROVINCE = "河北"`。

### 月度

原始数据放入 `data/original_data/`（销量 Excel + 温度子目录），然后：

```bash
uv run scripts/process_all.py
```

输出 `data/processed_data/{省份}.xlsx`。时间列为 `month`（`YYYY-MM`），其余气象列含义与旬度类似（按月聚合）。

切换省份：修改 `forecast_common.py` 中 `PROCESSED_PROVINCE = "河北"`。

---

## 特征工程

### 旬度 — `feature_engineering1.py`

`build_features_pipeline()` 在 processed 表上生成：

1. `temp_range`
2. `time_index`、`month_sin/cos`、`tenday_in_month`、`is_heating_season`、`spring_rework_peak`
3. `Lag_36`
4. `HDD_squared`、`HDD_cross_Lag_36`、`HDD_cross_HeatingSeason`、`ColdDays_cross_Lag_36`

因 `Lag_36` 会丢弃头部 36 行。

### 月度 — `feature_engineering.py`

`build_features_pipeline()` 生成：

1. `temp_range`
2. `time_index`、`month_sin/cos`、`is_heating_season`
3. `Lag_12`
4. `HDD_squared`、`HDD_cross_Lag_12`、`HDD_cross_HeatingSeason`、`ColdDays_cross_Lag_12`

因 `Lag_12` 会丢弃头部 12 行。无 `spring_rework_peak`、`tenday_in_month`。

---

## 运行模型

统一使用 `uv run` 执行

### 旬度 `scripts1/`

**表格特征 + 监督学习**


| 脚本                     | 模型                        |
| ---------------------- | ------------------------- |
| `run_lightgbm.py`      | LightGBM                  |
| `run_xgboost.py`       | XGBoost                   |
| `run_catboost.py`      | CatBoost                  |
| `run_random_forest.py` | 随机森林                      |
| `run_ridge.py`         | Ridge                     |
| `run_lasso.py`         | Lasso                     |
| `run_ElasticNet.py`    | ElasticNet                |
| `run_lstm.py`          | LSTM（连续特征 + 月份 Embedding） |


```bash
uv run scripts1/run_lightgbm.py
uv run scripts1/run_catboost.py
```

**时序 / 基础模型**


| 脚本                 | 说明                    |
| ------------------ | --------------------- |
| `run_prophet_0.py` | Prophet + XGBoost 残差  |
| `run_prophet_1.py` | Prophet + LightGBM 残差 |
| `run_sarimax.py`   | SARIMAX               |
| `run_chronus.py`   | Chronos               |
| `run_timesfm.py`   | TimesFM               |


输出目录：`output1/`。

### 月度 `scripts/`


| 脚本                                                           | 模型 / 用途         |
| ------------------------------------------------------------ | --------------- |
| `run_lightgbm_0.py`                                          | LightGBM        |
| `run_xgboost_0.py`                                           | XGBoost         |
| `run_catboost_0.py`                                          | CatBoost        |
| `run_random_forest_0.py`                                     | 随机森林            |
| `run_ridge_0.py`                                             | Ridge           |
| `run_lasso_0.py`                                             | Lasso           |
| `run_ElasticNet.py`                                          | ElasticNet      |
| `run_lstm_0.py`                                              | LSTM            |
| `run_gru_0.py`                                               | GRU             |
| `run_svr_0.py`                                               | SVR             |
| `run_nhits_0.py`                                             | N-HiTS          |
| `run_tide_0.py`                                              | TiDE            |
| `run_prophet_0.py` / `run_prophet_1.py`                      | Prophet 两阶段     |
| `run_sarimax_0.py`                                           | SARIMAX         |
| `run_chronus_0.py`                                           | Chronos         |
| `run_timesfm_0.py` / `run_timesfm_1.py` / `run_timesfm_2.py` | TimesFM         |
| `run_timemoe_0.py` / `run_timemoe_1.py`                      | TimeMoE         |
| `train_timemoe_peft.py` / `train_timemoe_peft_1.py`          | TimeMoE LoRA 训练 |
| `time_series_analysis.py`                                    | 时序分析            |
| `plot_test_result_0.py`                                      | 结果绘图            |


```bash
uv run scripts/run_lightgbm_0.py
uv run scripts/run_prophet_0.py
```

输出目录：`output/`。

---

## 训练切分约定

两套流程验证窗口语义一致，仅时间格式不同：


| 划分   | 旬度                          | 月度                    |
| ---- | --------------------------- | --------------------- |
| 训练截止 | `date <= 2025-06-21`        | `month <= 2025-06`    |
| 测试区间 | `2025-11-01` ~ `2026-03-21` | `2025-11` ~ `2026-03` |


全局配置分别在 `forecast_common1.py` 与 `forecast_common.py` 顶部；各 `run_*.py` 文件内也有 `TRAIN_END` / `TEST_START` / `TEST_END` 可单独调整。

---

## 输出说明

**旬度 `output1/`**

- `{省}_{模型}_test_{开始}_to_{结束}.xlsx`
- `{模型}/` 下特征重要性图、网格搜索 CSV
- `model_comparison/` 多模型对比

**月度 `output/`**

- 同上命名风格，月份为 `YYYY-MM`
- 各模型子目录（如 `output/lightgbm/`）

---

## 依赖概览

完整列表见 `pyproject.toml`，主要包括：

- 数据处理：`pandas`、`numpy`、`openpyxl`
- 机器学习：`scikit-learn`、`lightgbm`、`xgboost`、`catboost`
- 时序：`prophet`、`statsmodels`、`pmdarima`、`neuralforecast`、`timesfm`
- 深度学习：`torch`、`transformers`、`peft`、`accelerate`
- 可视化：`matplotlib`

---

## 常见问题

`**未找到 processed 文件**`

- 旬度：确认 `data/processed_data1/{省}.xlsx` 存在，或先跑 `process_sales_tenday.py`
- 月度：确认 `data/processed_data/{省}.xlsx` 存在，或先跑 `process_all.py`

`**ImportError: gas_prediction**`

先执行 `uv sync`；用 `uv run scripts1/xxx.py` 运行，不要直接用系统 Python。

**旬度特征工程报缺少春节日期**

在 `feature_engineering1.py` 的 `SPRING_FESTIVAL_DATES` 中补充对应年份。

**月度/旬度脚本混用报错**

检查 import 是否成对：`scripts1` 应对 `feature_engineering1` + `forecast_common1` + `processed_data1`。