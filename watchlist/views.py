from django.shortcuts import render, redirect, get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Stock
from .forms import StockForm
import yfinance as yf
import plotly.graph_objs as go
import plotly
import json
from django.utils.safestring import mark_safe
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin
from django.contrib import messages





@login_required
def home(request):
    form = StockForm(request.POST or None)
    if form.is_valid():
        new_stock = form.save(commit=False)
        new_stock.user = request.user
        new_stock.save()
        form = StockForm()

    stocks = Stock.objects.filter(user=request.user)
    stock_data = []

    for stock in stocks:
        try:
            ticker = yf.Ticker(stock.name)
            hist = ticker.history(period='7d')
            todays_data = ticker.history(period='1d')
            price = todays_data['Close'].iloc[-1] if not todays_data.empty else None
            prev_close = ticker.info.get('previousClose')

            if price is None or prev_close is None:
                continue

            change_percent = ((price - prev_close) / prev_close) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=stock.name))
            fig.update_layout(title=stock.name, margin=dict(l=0, r=0, t=30, b=0), height=250)
            chart = mark_safe(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

            stock_data.append({
                'id': stock.id,
                'name': stock.name,
                'price': round(price, 2),
                'change_percent': round(change_percent, 2),
                'chart': chart
            })

        except Exception:
            continue

    return render(request, 'home.html', {'stock_data': stock_data, 'form': form})


@login_required
def delete_stock(request, stock_id):
    stock = get_object_or_404(Stock, id=stock_id, user=request.user)
    stock.delete()
    return redirect('home')


@api_view(['GET', 'POST'])
@login_required
def stock_api(request):
    if request.method == 'POST':
        name = request.data.get('name')

        if not name:
            return Response({'error': 'No stock name provided'}, status=400)

        name = name.upper().strip()

        try:
            ticker = yf.Ticker(name)
            data = ticker.history(period='1d')

            if data.empty or 'Close' not in data.columns:
                return Response({'error': f'{name} is not a valid stock ticker or has no price data.'}, status=400)

            if Stock.objects.filter(name=name, user=request.user).exists():
                return Response({'error': f'{name} is already in your watchlist.'}, status=400)

            Stock.objects.create(name=name, user=request.user)
            return Response({'message': f'Stock {name} added successfully!'})

        except Exception as e:
            return Response({'error': f'Validation error: {str(e)}'}, status=500)

    stocks = Stock.objects.filter(user=request.user)
    stock_data = []

    for stock in stocks:
        try:
            ticker = yf.Ticker(stock.name)
            data = ticker.history(period='1d')
            price = data['Close'].iloc[-1] if not data.empty else None
            prev_close = ticker.info.get('previousClose')

            if price is None or prev_close is None:
                continue

            change_percent = ((price - prev_close) / prev_close) * 100

            stock_data.append({
                'id': stock.id,
                'name': stock.name,
                'price': round(price, 2),
                'change_percent': round(change_percent, 2),
            })

        except Exception:
            continue

    return Response(stock_data)


@login_required
def predict_stock(request):
    predicted_price = None
    ticker_input = ''
    test_days = 7
    feature_contributions = {}
    chart_base64 = None
    mae = rmse = None
    model_name = 'linear'
    selected_features = []
    best_model_chart = None  # ✅ Needed to prevent UnboundLocalError


    forecast_options = [7, 10, 15, 20, 30, 45, 60]
    feature_list = ['Day', 'MA3', 'Return', 'Volume_Change']

    comparison_results = []
    # 🔍 Find best model based on RMSE
    best_model_name = None
    if comparison_results:
        best_model = min(comparison_results, key=lambda x: x['rmse'])
        best_model_name = best_model['model']

    auto_tune = request.POST.get('auto_tune', 'off') == 'on' if request.method == 'POST' else False

    if request.method == 'POST':
        ticker_input = request.POST.get('ticker', '').upper().strip()
        test_days = int(request.POST.get('test_days', 7))
        model_name = request.POST.get('model_name', 'linear')
        selected_features = request.POST.getlist('features') or feature_list

        try:
            ticker = yf.Ticker(ticker_input)
            data = ticker.history(period='60d')

            if data.empty or 'Close' not in data.columns:
                raise ValueError("No valid data for this stock")

            df = data[['Close', 'Volume']].copy()
            df['MA3'] = df['Close'].rolling(window=3).mean()
            df['Return'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            df['Day'] = range(len(df))

            X = df[selected_features]
            y = df['Close']

            if len(X) <= test_days:
                raise ValueError(f"Not enough data to forecast {test_days} days. Try fewer days.")

            X_train, X_test = X[:-test_days], X[-test_days:]
            y_train, y_test = y[:-test_days], y[-test_days:]

            if model_name == 'decision_tree':
                model = DecisionTreeRegressor()
            elif model_name == 'random_forest':
                model = RandomForestRegressor(n_estimators=100)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            predicted_price = round(y_pred[-1], 2)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            # Model Comparison Table
            comparison_results = []
            for label, m in [
                ('Linear Regression', LinearRegression()),
                ('Decision Tree', DecisionTreeRegressor()),
                ('Random Forest', RandomForestRegressor(n_estimators=100))
            ]:
                m.fit(X_train, y_train)
                pred = m.predict(X_test)
                mae_val = mean_absolute_error(y_test, pred)
                rmse_val = mean_squared_error(y_test, pred, squared=False)

                comparison_results.append({
                    'model': label,
                    'mae': round(mae_val, 2),
                    'rmse': round(rmse_val, 2)
                })
            # 📊 Step 4: Identify the best model by RMSE and generate chart
            best_model_info = min(comparison_results, key=lambda x: x['rmse'])
            best_model_name = best_model_info['model']
            
            # Save prediction to database
            if request.user.is_authenticated:
                Stock.objects.create(
                    user=request.user,
                    ticker=ticker,
                    model_used=best_model_name,
                    forecast_days=int(test_days),
                    features_used=", ".join(selected_features),
                    predicted_price=predicted_price,
                    mae=mae,
                    rmse=rmse,
                )


            # Re-train best model
            if best_model_name == 'Linear Regression':
                best_model = LinearRegression()
            elif best_model_name == 'Decision Tree':
                best_model = DecisionTreeRegressor()
            elif best_model_name == 'Random Forest':
                best_model = RandomForestRegressor(n_estimators=100)

            best_model.fit(X_train, y_train)
            best_pred = best_model.predict(X_test)

            # Generate chart for best model
            plt.figure(figsize=(10, 4))
            x_col = X.columns.get_loc('Day') if 'Day' in X.columns else None
            x_train_vals = X_train.iloc[:, x_col] if x_col is not None else X_train.index
            x_test_vals = X_test.iloc[:, x_col] if x_col is not None else X_test.index

            plt.plot(x_train_vals, y_train, label="Train")
            plt.plot(x_test_vals, y_test, label="Actual", color="orange")
            plt.plot(x_test_vals, best_pred, label=f"{best_model_name} Forecast", linestyle="--", color="green")
            plt.title(f"Best Model: {best_model_name} Forecast")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            best_model_chart = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()


            # Feature Contributions (only Linear Regression)
            if hasattr(model, 'coef_'):
                last_row = X_test.iloc[-1]
                contributions = model.coef_ * last_row
                feature_contributions = {
                    f: round(c, 2) for f, c in zip(X.columns, contributions)
                }

            # Chart
            plt.figure(figsize=(10, 4))
            x_col = X.columns.get_loc('Day') if 'Day' in X.columns else None
            x_train_vals = X_train.iloc[:, x_col] if x_col is not None else X_train.index
            x_test_vals = X_test.iloc[:, x_col] if x_col is not None else X_test.index

            plt.plot(x_train_vals, y_train, label="Train")
            plt.plot(x_test_vals, y_test, label="Test", color="orange")
            plt.plot(x_test_vals, y_pred, label="Forecast", linestyle="--", color="green")
            plt.title(f"{ticker_input} - Train/Test & Forecast")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()

        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    else:
        selected_features = feature_list

    # ✅ Grid Search (Step 2 of Day 9)
    grid_results = []
    if request.method == 'POST' and model_name == 'random_forest' and auto_tune:
        for n in [10, 50, 100, 200]:
            rf = RandomForestRegressor(n_estimators=n)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            grid_results.append({
                'n_estimators': n,
                'mae': round(mean_absolute_error(y_test, pred), 2),
                'rmse': round(mean_squared_error(y_test, pred, squared=False), 2)
            })
        grid_results.sort(key=lambda x: x['rmse'])

    return render(request, 'predict.html', {
        'predicted_price': predicted_price,
        'ticker_input': ticker_input,
        'feature_contributions': feature_contributions,
        'chart': chart_base64,
        'mae': mae,
        'rmse': rmse,
        'test_days': test_days,
        'forecast_options': forecast_options,
        'feature_list': feature_list,
        'selected_features': selected_features,
        'model_name': model_name,
        'comparison_results': comparison_results,
        'grid_results': grid_results,
        'auto_tune': auto_tune,
        'best_model_name': best_model_name,
        'best_model_chart': best_model_chart if request.method == 'POST' else None,
        'best_model_name': best_model_name if request.method == 'POST' else None,
    })

@login_required
def history_view(request):
    user_predictions = Stock.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'predictions': user_predictions})

@login_required
def delete_history(request):
    if request.method == 'POST':
        Stock.objects.filter(user=request.user).delete()
        messages.success(request, "Prediction history deleted successfully.")
    return redirect('history')


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def debug_prediction_view(request):
    symbol = "RELIANCE.NS"
    df = yf.Ticker(symbol).history(period='30d')[['Close']]
    df = df.reset_index()
    df['Day'] = np.arange(len(df))

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    model = LinearRegression()
    model.fit(train[['Day']], train['Close'])

    test_preds = model.predict(test[['Day']])
    mae = mean_absolute_error(test['Close'], test_preds)
    rmse = mean_squared_error(test['Close'], test_preds, squared=False)

    plt.figure(figsize=(10, 5))
    plt.plot(train['Day'], train['Close'], label='Train')
    plt.plot(test['Day'], test['Close'], label='Test')
    plt.plot(test['Day'], test_preds, label='Forecast', linestyle='--')
    plt.title(f"{symbol} - Train/Test & Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render(request, 'debug_plot.html', {
        'chart': chart_data,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        
    })
