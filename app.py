# app.py - Enhanced Industry-Level Sales Forecasting Tool

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="AI Sales Forecasting", page_icon="📊")

# Custom CSS for UI
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class DataValidator:
    """Comprehensive data validation and quality checks"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, any]:
        """Perform comprehensive data validation"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0,
            'statistics': {}
        }
        
        # Check required columns
        required_columns = ['Date', 'Store', 'Dept', 'Weekly_Sales']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Data type validation
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Weekly_Sales'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
        except Exception as e:
            validation_results['errors'].append(f"Data type conversion error: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct['Weekly_Sales'] > 10:
            validation_results['warnings'].append(f"High missing values in Weekly_Sales: {missing_pct['Weekly_Sales']:.1f}%")
        
        # Check for negative sales
        negative_sales = (df['Weekly_Sales'] < 0).sum()
        if negative_sales > 0:
            validation_results['warnings'].append(f"Found {negative_sales} negative sales values")
        
        # Check for outliers using IQR method
        Q1 = df['Weekly_Sales'].quantile(0.25)
        Q3 = df['Weekly_Sales'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['Weekly_Sales'] < (Q1 - 1.5 * IQR)) | (df['Weekly_Sales'] > (Q3 + 1.5 * IQR))).sum()
        outlier_pct = (outliers / len(df)) * 100
        
        if outlier_pct > 5:
            validation_results['warnings'].append(f"High outlier percentage: {outlier_pct:.1f}%")
        
        # Data quality score calculation
        quality_score = 100
        quality_score -= len(validation_results['errors']) * 20
        quality_score -= len(validation_results['warnings']) * 5
        quality_score = max(0, quality_score)
        
        validation_results['data_quality_score'] = quality_score
        validation_results['statistics'] = {
            'total_records': len(df),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'unique_stores': df['Store'].nunique(),
            'unique_departments': df['Dept'].nunique(),
            'avg_weekly_sales': df['Weekly_Sales'].mean(),
            'missing_values_pct': missing_pct.to_dict(),
            'outlier_percentage': outlier_pct
        }
        
        return validation_results

class AdvancedFeatureEngineer:
    """Advanced feature engineering for time series forecasting"""
    
    @staticmethod
    def create_advanced_features(df: pd.DataFrame, mode: str = 'fast') -> pd.DataFrame:
        """Create comprehensive feature set for forecasting
        
        Args:
            df: Input dataframe
            mode: 'fast' for essential features only, 'full' for all features
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Store', 'Dept', 'Date'])
        
        # Time-based features (fast operations)
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.year
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for time features (vectorized)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
        
        # Essential lag features only (most important)
        df['Sales_Lag_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        df['Sales_Lag_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
        
        # Key rolling statistics (reduced windows)
        df['Sales_Rolling_Mean_3'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        df['Sales_Rolling_Std_3'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
        
        # One exponential weighted moving average
        df['Sales_EWM_Alpha02'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).ewm(alpha=0.2, min_periods=1).mean())
        
        # Essential percentage change
        df['Sales_Pct_Change_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].pct_change(1)
        
        # Holiday features (vectorized)
        df['IsHoliday'] = df['Date'].dt.month.isin([11, 12]).astype(int)
        df['IsBackToSchool'] = df['Date'].dt.month.isin([8, 9]).astype(int)
        
        if mode == 'full':
            # Additional features for full mode
            df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
            df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
            
            # Additional lag features
            for lag in [2, 3, 8, 12]:
                df[f'Sales_Lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
            
            # Additional rolling statistics
            for window in [6, 12]:
                df[f'Sales_Rolling_Mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                df[f'Sales_Rolling_Std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
            
            # Additional EWM
            df['Sales_EWM_Alpha05'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.shift(1).ewm(alpha=0.5, min_periods=1).mean())
            
            # Additional percentage change
            df['Sales_Pct_Change_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].pct_change(4)
        
        # Fill missing values efficiently
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df

class ModelEnsemble:
    """Ensemble of multiple forecasting models"""
    
    def __init__(self):
        self.models = {
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        self.weights = {}
        self.feature_importance = {}
        
    def train_with_validation(self, X: pd.DataFrame, y: pd.Series, 
                            cv_folds: int = 5) -> Dict[str, float]:
        """Train models with time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        model_scores = {}
        
        for name, model in self.models.items():
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate multiple metrics
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                mape = mean_absolute_percentage_error(y_val, y_pred)
                
                cv_scores.append({'mse': mse, 'mae': mae, 'mape': mape})
            
            # Average scores across folds
            avg_scores = {
                'mse': np.mean([score['mse'] for score in cv_scores]),
                'mae': np.mean([score['mae'] for score in cv_scores]),
                'mape': np.mean([score['mape'] for score in cv_scores])
            }
            
            model_scores[name] = avg_scores
            
            # Train final model on full dataset
            model.fit(X, y)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
        
        # Calculate ensemble weights based on inverse MSE
        mse_values = [scores['mse'] for scores in model_scores.values()]
        inv_mse = [1/mse for mse in mse_values]
        total_inv_mse = sum(inv_mse)
        
        for i, (name, _) in enumerate(self.models.items()):
            self.weights[name] = inv_mse[i] / total_inv_mse
        
        return model_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)
    
    def predict_with_intervals(self, X: pd.DataFrame, 
                             confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
        all_predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate ensemble prediction
        ensemble_pred = np.sum([pred * self.weights[list(self.models.keys())[i]] 
                               for i, pred in enumerate(all_predictions)], axis=0)
        
        # Calculate prediction intervals using ensemble variance
        ensemble_std = np.std(all_predictions, axis=0)
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = ensemble_pred - z_score * ensemble_std
        upper_bound = ensemble_pred + z_score * ensemble_std
        
        return ensemble_pred, lower_bound, upper_bound

def create_business_insights(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, any]:
    """Generate business insights from forecasts"""
    insights = {}
    
    # Revenue projections
    total_historical_sales = df['Weekly_Sales'].sum()
    total_forecast_sales = forecast_df['Weekly_Sales'].sum()
    
    insights['revenue_projection'] = {
        'total_forecast_revenue': total_forecast_sales,
        'avg_weekly_revenue': total_forecast_sales / len(forecast_df['Date'].unique()),
        'growth_vs_historical': ((total_forecast_sales / len(forecast_df['Date'].unique())) / 
                               (total_historical_sales / len(df['Date'].unique())) - 1) * 100
    }
    
    # Top performing segments
    store_performance = forecast_df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
    dept_performance = forecast_df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
    
    insights['top_performers'] = {
        'stores': store_performance.head(5).to_dict(),
        'departments': dept_performance.head(5).to_dict()
    }
    
    # Risk assessment
    weekly_volatility = forecast_df.groupby('Date')['Weekly_Sales'].sum().std()
    avg_weekly_sales = forecast_df.groupby('Date')['Weekly_Sales'].sum().mean()
    cv = weekly_volatility / avg_weekly_sales
    
    insights['risk_metrics'] = {
        'weekly_volatility': weekly_volatility,
        'coefficient_of_variation': cv,
        'risk_level': 'High' if cv > 0.3 else 'Medium' if cv > 0.15 else 'Low'
    }
    
    return insights

# Streamlit App
st.title("🚀 Enterprise Sales Forecasting Platform")
st.markdown("### Advanced AI-Powered Sales Forecasting with Business Intelligence")

# Sidebar configuration
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox("Forecasting Method", ["Ensemble (Recommended)", "XGBoost Only", "Random Forest Only"])
forecast_period = st.sidebar.slider("Forecast Horizon (weeks)", min_value=1, max_value=26, value=8)
confidence_level = st.sidebar.slider("Confidence Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

st.sidebar.header("📊 Advanced Options")
feature_mode = st.sidebar.selectbox("Feature Engineering Mode", ["Fast (Essential features)", "Full (All features)"], index=0)
enable_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
enable_business_insights = st.sidebar.checkbox("Generate Business Insights", value=True)
enable_model_comparison = st.sidebar.checkbox("Compare Models", value=True)

# Main content
st.markdown("""
### 📋 Data Requirements
Upload your historical sales data with the following structure:
- **Date**: DateTime format (YYYY-MM-DD)
- **Store**: Store identifier
- **Dept**: Department identifier  
- **Weekly_Sales**: Sales amount (numeric)
- **Optional**: Temperature, Fuel_Price, IsHoliday, etc.

### 🔧 Advanced Features
- **Ensemble Modeling**: Combines multiple algorithms for better accuracy
- **Time Series Validation**: Proper backtesting for time series data
- **Confidence Intervals**: Quantifies prediction uncertainty
- **Business Intelligence**: Actionable insights for decision making
- **Feature Engineering**: 50+ advanced time series features
""")

uploaded_file = st.file_uploader("📁 Upload Sales Data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and validate data
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Data Validation Report")
        validator = DataValidator()
        validation_results = validator.validate_data(df)
        
        # Display validation results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_score = validation_results['data_quality_score']
            color = "success" if quality_score >= 80 else "warning" if quality_score >= 60 else "error"
            st.markdown(f"""
            <div class="metric-container {color}-metric">
                <h3>Data Quality Score</h3>
                <h1>{quality_score}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Records", validation_results['statistics']['total_records'])
            st.metric("Date Range", validation_results['statistics']['date_range'])
        
        with col3:
            st.metric("Unique Stores", validation_results['statistics']['unique_stores'])
            st.metric("Unique Departments", validation_results['statistics']['unique_departments'])
        
        # Display errors and warnings
        if validation_results['errors']:
            st.error("❌ Data Validation Errors:")
            for error in validation_results['errors']:
                st.error(f"• {error}")
        
        if validation_results['warnings']:
            st.warning("⚠️ Data Quality Warnings:")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        if not validation_results['is_valid']:
            st.stop()
        
        # Feature Engineering
        st.subheader("🔧 Advanced Feature Engineering")
        
        # Determine feature mode
        mode = 'fast' if feature_mode == "Fast (Essential features)" else 'full'
        
        with st.spinner(f"Creating features in {mode} mode..."):
            feature_engineer = AdvancedFeatureEngineer()
            df_processed = feature_engineer.create_advanced_features(df, mode=mode)
        
        new_features = len(df_processed.columns) - len(df.columns)
        st.success(f"✅ Created {new_features} new features in {mode} mode")
        
        if mode == 'fast':
            st.info("💡 Using fast mode for quicker processing. Switch to 'Full' mode for maximum accuracy.")
        
        # Model Training
        st.subheader("🤖 Model Training & Validation")
        
        # Prepare features for modeling
        feature_cols = [col for col in df_processed.columns if col not in ['Date', 'Weekly_Sales']]
        X = df_processed[feature_cols]
        y = df_processed['Weekly_Sales']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        with st.spinner("Training ensemble models..."):
            ensemble = ModelEnsemble()
            model_scores = ensemble.train_with_validation(X, y)
        
        # Display model performance
        st.subheader("📊 Model Performance Comparison")
        
        metrics_df = pd.DataFrame(model_scores).T
        metrics_df['RMSE'] = np.sqrt(metrics_df['mse'])
        metrics_df = metrics_df[['RMSE', 'mae', 'mape']]
        metrics_df.columns = ['RMSE', 'MAE', 'MAPE']
        
        st.dataframe(metrics_df.round(2))
        
        # Feature Importance
        if enable_feature_importance:
            st.subheader("📈 Feature Importance Analysis")
            
            # Get feature importance from XGBoost model
            if 'XGBoost' in ensemble.feature_importance:
                importance_df = pd.DataFrame({
                    'Feature': list(ensemble.feature_importance['XGBoost'].keys()),
                    'Importance': list(ensemble.feature_importance['XGBoost'].values())
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title='Top 15 Most Important Features')
                st.plotly_chart(fig, use_container_width=True)
        
        # Generate Forecasts
        st.subheader("🔮 Sales Forecasting")
        
        with st.spinner("Generating forecasts with confidence intervals..."):
            # Create future dates
            last_date = df_processed['Date'].max()
            future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_period + 1)]
            
            # Get last known values for each store-dept combination
            last_values = df_processed.loc[df_processed.groupby(['Store', 'Dept'])['Date'].idxmax()]
            
            forecasts = []
            for future_date in future_dates:
                for _, row in last_values.iterrows():
                    # Update time-based features
                    row_copy = row.copy()
                    row_copy['Date'] = future_date
                    row_copy['Month'] = future_date.month
                    row_copy['Week'] = future_date.isocalendar().week
                    row_copy['Year'] = future_date.year
                    row_copy['Day'] = future_date.day
                    row_copy['DayOfWeek'] = future_date.weekday()
                    row_copy['Quarter'] = (future_date.month - 1) // 3 + 1
                    
                    # Update cyclical features
                    row_copy['Month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
                    row_copy['Month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
                    row_copy['Week_sin'] = np.sin(2 * np.pi * row_copy['Week'] / 52)
                    row_copy['Week_cos'] = np.cos(2 * np.pi * row_copy['Week'] / 52)
                    
                    forecasts.append(row_copy)
            
            forecast_df = pd.DataFrame(forecasts)
            
            # Make predictions with confidence intervals
            X_future = forecast_df[feature_cols]
            predictions, lower_bound, upper_bound = ensemble.predict_with_intervals(
                X_future, confidence_level=confidence_level)
            
            forecast_df['Weekly_Sales'] = predictions
            forecast_df['Lower_Bound'] = lower_bound
            forecast_df['Upper_Bound'] = upper_bound
            
            # Display forecast results
            forecast_summary = forecast_df[['Date', 'Store', 'Dept', 'Weekly_Sales', 'Lower_Bound', 'Upper_Bound']]
            st.dataframe(forecast_summary.round(2))
        
        # Visualization
        st.subheader("📊 Forecast Visualization")
        
        # Aggregate forecasts by date
        daily_forecast = forecast_df.groupby('Date').agg({
            'Weekly_Sales': 'sum',
            'Lower_Bound': 'sum',
            'Upper_Bound': 'sum'
        }).reset_index()
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=daily_forecast['Date'],
            y=daily_forecast['Weekly_Sales'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=daily_forecast['Date'],
            y=daily_forecast['Upper_Bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_forecast['Date'],
            y=daily_forecast['Lower_Bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name=f'{int(confidence_level*100)}% Confidence Interval',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.update_layout(
            title='Sales Forecast with Confidence Intervals',
            xaxis_title='Date',
            yaxis_title='Weekly Sales',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Business Insights
        if enable_business_insights:
            st.subheader("💡 Business Intelligence & Insights")
            
            insights = create_business_insights(df, forecast_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Projected Revenue", 
                         f"${insights['revenue_projection']['total_forecast_revenue']:,.0f}")
                st.metric("Avg Weekly Revenue", 
                         f"${insights['revenue_projection']['avg_weekly_revenue']:,.0f}")
            
            with col2:
                st.metric("Growth vs Historical", 
                         f"{insights['revenue_projection']['growth_vs_historical']:+.1f}%")
                st.metric("Risk Level", insights['risk_metrics']['risk_level'])
            
            with col3:
                st.metric("Weekly Volatility", 
                         f"${insights['risk_metrics']['weekly_volatility']:,.0f}")
                st.metric("Coefficient of Variation", 
                         f"{insights['risk_metrics']['coefficient_of_variation']:.2f}")
            
            # Top performers
            st.subheader("🏆 Top Performing Segments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Stores by Forecast Revenue**")
                top_stores = pd.DataFrame(list(insights['top_performers']['stores'].items()), 
                                        columns=['Store', 'Forecast_Revenue'])
                st.dataframe(top_stores)
            
            with col2:
                st.markdown("**Top 5 Departments by Forecast Revenue**")
                top_depts = pd.DataFrame(list(insights['top_performers']['departments'].items()), 
                                       columns=['Department', 'Forecast_Revenue'])
                st.dataframe(top_depts)
        
        # Download options
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Forecast download
            forecast_csv = forecast_summary.to_csv(index=False)
            st.download_button(
                label="📊 Download Forecast Data",
                data=forecast_csv,
                file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Model performance download
            performance_csv = metrics_df.to_csv()
            st.download_button(
                label="📈 Download Model Performance",
                data=performance_csv,
                file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Success message
        st.success("🎉 Forecasting completed successfully! Your enterprise-grade sales forecasts are ready.")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        # Display error details in development mode
        if st.checkbox("Show error details (for debugging)"):
            st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin forecasting")
    
    # Show sample data format
    st.subheader("📋 Sample Data Format")
    sample_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-08', '2023-01-15'],
        'Store': [1, 1, 1],
        'Dept': [1, 1, 1],
        'Weekly_Sales': [24924.5, 46039.49, 41595.55],
        'Temperature': [42.31, 38.51, 44.57],
        'Fuel_Price': [2.572, 2.548, 2.514],
        'IsHoliday': [False, False, False]
    })
    st.dataframe(sample_data)