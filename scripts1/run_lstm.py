import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from gas_prediction.feature_engineering1 import build_features_pipeline
from gas_prediction.forecast_common1 import (
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

warnings.filterwarnings("ignore")

# 旬度时间切分：训练集 date<=TRAIN_END；测试集 [TEST_START, TEST_END] 闭区间（旬起始日 YYYY-MM-DD）。
TRAIN_END = "2025-06-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"

OUTPUT_ROOT = "output1"

# 旬度下原月度 Lag_12 对应为 Lag_36（约一年前的同一旬）；序列长度 9≈3 个月×3 旬/月，与原先「看 3 个月」同量级。
SELECTED_FEATURES = [
    "Lag_36",
    "HDD",
    "is_heating_season",
]

SEQ_LENGTH = 9
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 0.005
EPOCHS = 300
BATCH_SIZE = 32
RANDOM_SEED = 42
EMBEDDING_DIM = 4


class GasForecastLSTM(nn.Module):
    def __init__(self, continuous_size: int, hidden_size: int, num_layers: int, emb_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.month_emb = nn.Embedding(num_embeddings=13, embedding_dim=emb_dim)

        total_input_size = continuous_size + emb_dim

        self.lstm = nn.LSTM(
            input_size=total_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_cont: torch.Tensor, x_month: torch.Tensor) -> torch.Tensor:
        month_vecs = self.month_emb(x_month)
        x = torch.cat((x_cont, month_vecs), dim=-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        return self.fc(out[:, -1, :])


def _load_and_build_features() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    }
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    model_input_cols = [
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    ]
    raw_df = raw_df[model_input_cols].copy()
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df["HDD"] = pd.to_numeric(raw_df["HDD"], errors="coerce")
    raw_df["extreme_cold_days"] = pd.to_numeric(raw_df["extreme_cold_days"], errors="coerce")
    raw_df = raw_df.dropna(subset=list(required)).reset_index(drop=True)

    df_features = build_features_pipeline(raw_df, target_col=TARGET_COL, date_col=DATE_COL)
    df_features["month_idx"] = pd.to_datetime(df_features[DATE_COL]).dt.month.astype(int)
    return df_features


def _create_3d_sequences(
    x_cont_arr: np.ndarray,
    x_month_arr: np.ndarray,
    y_arr: np.ndarray,
    date_arr: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs_cont: list[np.ndarray] = []
    xs_month: list[np.ndarray] = []
    ys: list[float] = []
    ds: list[str] = []

    for i in range(len(x_cont_arr) - seq_length):
        xs_cont.append(x_cont_arr[i : i + seq_length])
        xs_month.append(x_month_arr[i : i + seq_length])
        ys.append(y_arr[i + seq_length])
        ds.append(str(date_arr[i + seq_length]))

    return np.asarray(xs_cont), np.asarray(xs_month), np.asarray(ys), np.asarray(ds)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE(%)": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    print("启动 PyTorch LSTM 旬度建模流程...")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df_features = _load_and_build_features().sort_values(DATE_COL).reset_index(drop=True)
    missing_features = [f for f in SELECTED_FEATURES if f not in df_features.columns]
    if missing_features:
        raise ValueError(f"特征工程后缺少所需特征: {missing_features}")

    date_arr = df_features[DATE_COL].astype(str).to_numpy()
    x_cont_all = df_features[SELECTED_FEATURES].to_numpy(dtype=float)
    x_month_all = df_features["month_idx"].to_numpy(dtype=int)
    y_all = df_features[TARGET_COL].to_numpy(dtype=float)

    train_mask = date_arr <= TRAIN_END
    if not np.any(train_mask):
        raise ValueError("训练集为空，请检查时间范围和数据。")

    test_mask_raw = (date_arr >= TEST_START) & (date_arr <= TEST_END)
    if not np.any(test_mask_raw):
        raise ValueError("测试集为空，请检查时间范围和数据。")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x_cont_all[train_mask])
    scaler_y.fit(y_all[train_mask].reshape(-1, 1))

    x_cont_scaled = scaler_x.transform(x_cont_all)
    y_scaled = scaler_y.transform(y_all.reshape(-1, 1)).flatten()

    x_cont_3d, x_month_3d, y_1d, y_date = _create_3d_sequences(
        x_cont_scaled, x_month_all, y_scaled, date_arr, SEQ_LENGTH
    )

    if len(x_cont_3d) == 0:
        raise ValueError("样本量不足，无法构造 LSTM 序列。")

    seq_train_mask = y_date <= TRAIN_END
    seq_test_mask = (y_date >= TEST_START) & (y_date <= TEST_END)

    x_cont_train = x_cont_3d[seq_train_mask]
    x_month_train = x_month_3d[seq_train_mask]
    y_train = y_1d[seq_train_mask]

    x_cont_test = x_cont_3d[seq_test_mask]
    x_month_test = x_month_3d[seq_test_mask]
    y_test_scaled = y_1d[seq_test_mask]
    y_test_date = y_date[seq_test_mask]

    x_cont_train_tensor = torch.tensor(x_cont_train, dtype=torch.float32)
    x_month_train_tensor = torch.tensor(x_month_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_cont_test_tensor = torch.tensor(x_cont_test, dtype=torch.float32)
    x_month_test_tensor = torch.tensor(x_month_test, dtype=torch.long)

    train_dataset = TensorDataset(x_cont_train_tensor, x_month_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GasForecastLSTM(
        continuous_size=len(SELECTED_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        emb_dim=EMBEDDING_DIM,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("开始训练 LSTM 网络...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x_cont, b_x_month, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x_cont, b_x_month)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(x_cont_test_tensor, x_month_test_tensor).cpu().numpy().flatten()

    test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    result_df = pd.DataFrame(
        {
            DATE_COL: y_test_date.astype(str),
            "actual_gas_sales": y_test.astype(float),
            "predicted_gas_sales": test_preds.astype(float),
        }
    )
    result_df["error"] = result_df["predicted_gas_sales"] - result_df["actual_gas_sales"]
    result_df["abs_error"] = np.abs(result_df["error"])
    result_df["mape_pct"] = np.where(
        result_df["actual_gas_sales"] != 0,
        np.abs(result_df["error"] / result_df["actual_gas_sales"]) * 100.0,
        np.nan,
    )

    metrics = _regression_metrics(y_test.astype(float), test_preds.astype(float))
    metrics_df = pd.DataFrame([metrics])

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    excel_path = os.path.join(
        OUTPUT_ROOT,
        f"{PROCESSED_PROVINCE}_lstm_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, index=False, sheet_name="test_forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")
    print(f"测试集预测结果已保存: {excel_path}")

    print(f"\nLSTM 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
