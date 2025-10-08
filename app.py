import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Any

from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
# Optional import kept from original code
from statsmodels.tsa.arima.model import ARIMA  # noqa: F401

from statsmodels.tools.sm_exceptions import ConvergenceWarning

# -----------------------------
# Config and warnings
# -----------------------------
st.set_page_config(page_title="Hybrid ARIMA + ML Forecast", layout="wide")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

FEATURES = ['Household_Income_RM', 'Lending_Rate', 'CPI', 'GDP_RM', 'Population_000']

MODELS = {
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())]),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'Polynomial Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('nn', MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(50,)))
    ]),
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_csv(file_or_path: Any):
    if hasattr(file_or_path, "read"):  # UploadedFile
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)
    # Clean
    if 'State' in df.columns:
        df['State'] = df['State'].astype(str).str.strip().str.upper()
    # Drop unnamed cols
    cols_to_drop = [c for c in df.columns if c.lower().startswith('unnamed')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if 'Year' in df.columns:
        # ensure numeric year
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    return df


def train_evaluate_and_forecast(df: pd.DataFrame, target_state: str, forecast_year: int, future_features_data: dict):
    # Filter and sort by year
    state_df = df[df['State'] == target_state].set_index('Year').sort_index()
    # Use a DatetimeIndex to silence statsmodels/pmdarima index warnings
    state_df.index = pd.to_datetime(state_df.index.astype(str), format="%Y", errors="coerce")
    state_df = state_df[~state_df.index.isna()]  # drop bad rows if any

    # Impute feature NaNs with mean
    for f in FEATURES:
        if f in state_df.columns and state_df[f].isnull().any():
            state_df[f] = state_df[f].fillna(state_df[f].mean())

    if state_df['Median_House_Price'].isnull().any():
        raise ValueError(f"Missing values in Median_House_Price for {target_state}")

    y = state_df['Median_House_Price']
    years = state_df.index.to_numpy()

    if len(years) < 5:
        raise ValueError("Insufficient data (need at least 5 years).")

    split_idx = int(len(years) * 0.8)
    train_years, test_years = years[:split_idx], years[split_idx:]

    y_train, y_test = y.loc[train_years], y.loc[test_years]
    X_train, X_test = state_df.loc[train_years, FEATURES], state_df.loc[test_years, FEATURES]

    # ARIMA on training target
    arima_model = auto_arima(
        y_train,
        seasonal=False,
        suppress_warnings=True,
        max_p=5, max_q=5,
        stepwise=True,
        error_action="ignore",
        trace=False
    )
    train_fitted = arima_model.predict_in_sample()
    train_residuals = y_train - train_fitted

    if np.allclose(train_residuals, 0) or np.isnan(train_residuals).any():
        raise ValueError("ARIMA residuals invalid (all zero or NaN).")

    # Evaluate ML models on residuals
    evaluation_results = {}
    for name, mdl in MODELS.items():
        mdl.fit(X_train, train_residuals)
        arima_test_pred = arima_model.predict(n_periods=len(test_years))
        ml_test_residual_pred = mdl.predict(X_test)
        hybrid_test_pred = np.asarray(arima_test_pred) + np.asarray(ml_test_residual_pred)

        mse = mean_squared_error(y_test, hybrid_test_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, hybrid_test_pred)
        evaluation_results[name] = {'MSE': float(mse), 'RMSE': rmse, 'MAE': float(mae)}

    best_model_name = min(evaluation_results, key=lambda k: evaluation_results[k]['MSE'])
    best_model = MODELS[best_model_name]

    # Refit on full data
    arima_full = auto_arima(
        y,
        seasonal=False,
        suppress_warnings=True,
        max_p=5, max_q=5,
        stepwise=True,
        error_action="ignore",
        trace=False
    )
    full_fitted = arima_full.predict_in_sample()
    full_residuals = y - full_fitted

    X_full = state_df[FEATURES]
    best_model.fit(X_full, full_residuals)

    # Forecast future year
    last_year = int(state_df.index.year.max())
    periods_ahead = int(forecast_year - last_year)
    if periods_ahead <= 0:
        raise ValueError(f"Forecast year must be > {last_year}")

    future_features = pd.DataFrame([future_features_data], index=[forecast_year])[FEATURES]
    arima_future_seq = arima_full.predict(n_periods=periods_ahead)
    arima_final_forecast = float(arima_future_seq[-1])
    ml_future = float(best_model.predict(future_features)[0])
    hybrid_forecast = max(0.0, arima_final_forecast + ml_future)

    return {
        "evaluation": evaluation_results,
        "best_model_name": best_model_name,
        "series_years": years,
        "series_values": y.values,
        "forecast_year": forecast_year,
        "hybrid_forecast": hybrid_forecast,
        "arima_component": arima_final_forecast,
        "ml_component": ml_future,
        "last_year": last_year,
        "target_state": target_state
    }


def plot_history_and_forecast(years, values, forecast_year, forecast_value, state):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, values, label='Historical Median House Price', color='blue', marker='o')
    ax.scatter([forecast_year], [forecast_value], label='Forecasted Price', color='red', s=80, zorder=3)
    ax.set_title(f'Median House Price Trend for {state}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Median House Price (RM)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ymax = max(np.max(values), forecast_value) * 1.2
    ax.set_ylim(bottom=0, top=ymax)
    ax.legend()
    fig.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("Hybrid Forecast: ARIMA + ML Residuals")

with st.sidebar:
    st.header("Data")
    st.caption("Upload your CSV or use a path in project.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    default_path = st.text_input("Or CSV path", value="Datathon_ML_Hybrid.csv")
    load_btn = st.button("Load Data")

df = None
if 'df' not in st.session_state:
    st.session_state.df = None

if load_btn:
    try:
        if uploaded is not None:
            st.session_state.df = load_data_from_csv(uploaded)
        else:
            st.session_state.df = load_data_from_csv(default_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")

df = st.session_state.df

if df is None:
    st.info("Load a CSV to begin. Expected columns include: State, Year, Median_House_Price and feature columns.")
    st.stop()

missing_cols = [c for c in ['State', 'Year', 'Median_House_Price', *FEATURES] if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

states = sorted(df['State'].dropna().unique().tolist())
col1, col2 = st.columns([1, 1])

with col1:
    target_state = st.selectbox("Select State", states, index=0)

state_df_preview = df[df['State'] == target_state].sort_values('Year')
last_year = int(state_df_preview['Year'].max())

with col2:
    forecast_year = st.number_input(
        f"Forecast Year (>{last_year})",
        min_value=last_year + 1, step=1, value=last_year + 1
    )

st.markdown("### Future Feature Inputs")
defaults = state_df_preview.set_index('Year').sort_index().iloc[-1][FEATURES].to_dict()

feat_cols = st.columns(len(FEATURES))
future_inputs = {}
for i, f in enumerate(FEATURES):
    min_val = 0.0 if f in ['Household_Income_RM', 'Population_000', 'GDP_RM'] else -1e9
    with feat_cols[i]:
        default_val = float(defaults.get(f, 0.0))
        future_inputs[f] = st.number_input(f, value=default_val, step=0.1, min_value=min_val)

run = st.button("Run Forecast")

if run:
    with st.spinner("Training models and generating forecast..."):
        try:
            result = train_evaluate_and_forecast(df, target_state, int(forecast_year), future_inputs)

            # Metrics table
            st.subheader("Model Evaluation (Test Set)")
            metrics_df = pd.DataFrame(result["evaluation"]).T.sort_values("MSE")
            st.dataframe(metrics_df.style.format({"MSE": "{:.2f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}), use_container_width=True)

            st.success(f"Best Model: {result['best_model_name']}")
            st.write(f"ARIMA Base Forecast ({forecast_year}): {result['arima_component']:,.2f}")
            st.write(f"ML Residual Adjustment: {result['ml_component']:,.2f}")
            st.markdown(f"**Final Hybrid Forecast ({target_state}, {forecast_year}): {result['hybrid_forecast']:,.2f}**")

            # Plot
            fig = plot_history_and_forecast(
                years=result["series_years"],
                values=result["series_values"],
                forecast_year=result["forecast_year"],
                forecast_value=result["hybrid_forecast"],
                state=result["target_state"]
            )
            st.pyplot(fig, clear_figure=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")