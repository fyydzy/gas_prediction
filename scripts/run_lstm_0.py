import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from gas_prediction.feature_engineering import build_features_pipeline
from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

warnings.filterwarnings("ignore")

# 训练集截止 2025-06，测试集为 2025-11~2026-03
TRAIN_END = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"

OUTPUT_DIR = "output"

# 极简黄金特征矩阵
SELECTED_FEATURES = [
    "Lag_12", 
    "HDD", 
    "is_heating_season"
]

SEQ_LENGTH = 3
HIDDEN_SIZE = 16
NUM_LAYERS = 1
LEARNING_RATE = 0.005
EPOCHS = 300
BATCH_SIZE = 16
RANDOM_SEED = 42
EMBEDDING_DIM = 4 

# ==========================================
# 核心替换：将 GRU 修改为 LSTM
# ==========================================
class GasForecastLSTM(nn.Module):
    def __init__(self, continuous_size: int, hidden_size: int, num_layers: int, emb_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 月份实体嵌入层
        self.month_emb = nn.Embedding(num_embeddings=13, embedding_dim=emb_dim)
        
        total_input_size = continuous_size + emb_dim
        
        # 替换为 nn.LSTM
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
        
        # LSTM 必须同时初始化隐藏状态 (h0) 和 细胞状态 (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # 将 (h0, c0) 作为元组传入
        out, _ = self.lstm(x, (h0, c0))
        
        return self.fc(out[:, -1, :])
# ==========================================


def _load_and_build_features() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    raw_df = raw_df.copy()
    raw_df[MONTH_COL] = raw_df[MONTH_COL].astype(str).str.slice(0, 7)
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df = raw_df.dropna(
        subset=[MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"]
    ).reset_index(drop=True)
    
    df_features = build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)
    df_features['month_idx'] = pd.to_datetime(df_features[MONTH_COL]).dt.month
    return df_features


def _create_3d_sequences(
    x_cont_arr: np.ndarray,
    x_month_arr: np.ndarray,
    y_arr: np.ndarray,
    month_arr: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs_cont: list[np.ndarray] = []
    xs_month: list[np.ndarray] = []
    ys: list[float] = []
    ms: list[str] = []
    
    for i in range(len(x_cont_arr) - seq_length):
        xs_cont.append(x_cont_arr[i : i + seq_length])
        xs_month.append(x_month_arr[i : i + seq_length])
        ys.append(y_arr[i + seq_length])
        ms.append(month_arr[i + seq_length])
        
    return np.asarray(xs_cont), np.asarray(xs_month), np.asarray(ys), np.asarray(ms)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def main() -> None:
    print("启动 PyTorch LSTM 真实数据建模流程...")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df_features = _load_and_build_features().sort_values(MONTH_COL).reset_index(drop=True)
    missing_features = [f for f in SELECTED_FEATURES if f not in df_features.columns]
    if missing_features:
        raise ValueError(f"特征工程后缺少所需特征: {missing_features}")

    month_arr = df_features[MONTH_COL].astype(str).to_numpy()
    x_cont_all = df_features[SELECTED_FEATURES].to_numpy(dtype=float)
    x_month_all = df_features['month_idx'].to_numpy(dtype=int) 
    y_all = df_features[TARGET_COL].to_numpy(dtype=float)

    train_mask = month_arr <= TRAIN_END
    if not np.any(train_mask):
        raise ValueError("训练集为空，请检查时间范围和数据。")

    test_mask_raw = (month_arr >= TEST_START) & (month_arr <= TEST_END)
    if not np.any(test_mask_raw):
        raise ValueError("测试集为空，请检查时间范围和数据。")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x_cont_all[train_mask])
    scaler_y.fit(y_all[train_mask].reshape(-1, 1))
    
    x_cont_scaled = scaler_x.transform(x_cont_all)
    y_scaled = scaler_y.transform(y_all.reshape(-1, 1)).flatten()

    x_cont_3d, x_month_3d, y_1d, y_month = _create_3d_sequences(
        x_cont_scaled, x_month_all, y_scaled, month_arr, SEQ_LENGTH
    )
    
    if len(x_cont_3d) == 0:
        raise ValueError("样本量不足，无法构造 LSTM 序列。")

    seq_train_mask = y_month <= TRAIN_END
    seq_test_mask = (y_month >= TEST_START) & (y_month <= TEST_END)

    x_cont_train = x_cont_3d[seq_train_mask]
    x_month_train = x_month_3d[seq_train_mask]
    y_train = y_1d[seq_train_mask]
    
    x_cont_test = x_cont_3d[seq_test_mask]
    x_month_test = x_month_3d[seq_test_mask]
    y_test_scaled = y_1d[seq_test_mask]
    y_test_month = y_month[seq_test_mask]

    x_cont_train_tensor = torch.tensor(x_cont_train, dtype=torch.float32)
    x_month_train_tensor = torch.tensor(x_month_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    x_cont_test_tensor = torch.tensor(x_cont_test, dtype=torch.float32)
    x_month_test_tensor = torch.tensor(x_month_test, dtype=torch.long)

    train_dataset = TensorDataset(x_cont_train_tensor, x_month_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 实例化 LSTM 模型
    model = GasForecastLSTM(
        continuous_size=len(SELECTED_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        emb_dim=EMBEDDING_DIM
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
            MONTH_COL: y_test_month.astype(str),
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 修改输出文件名为 LSTM
    excel_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_lstm_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")

    mape = _mape(y_test.astype(float), test_preds.astype(float))
    if np.isfinite(mape):
        print(f"\nLSTM 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":
    main()