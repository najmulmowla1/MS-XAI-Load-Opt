# Step 03: Common data and evaluation utilities

def print_section(title):
    print('\n' + '='*100)
    print(title)
    print('='*100)

def sanitize_column_name(col):
    col = str(col)
    col = re.sub(r'[^0-9a-zA-Z_]+', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

def sanitize_columns(df):
    out = df.copy()
    out.columns = [sanitize_column_name(c) for c in out.columns]
    return out

def parse_hourly_datetime(series):
    formats = [
        '%d:%m:%Y:%H:%M', '%d-%m-%y %H:%M', '%d-%m-%Y %H:%M',
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M'
    ]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors='coerce')
        if parsed.notna().mean() > 0.95:
            return parsed
    return pd.to_datetime(series, errors='coerce', dayfirst=True)

def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true-y_pred)/np.maximum(np.abs(y_true), eps))) * 100)

def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    den = np.maximum((np.abs(y_true)+np.abs(y_pred))/2, eps)
    return float(np.mean(np.abs(y_true-y_pred)/den) * 100)

def mase_scale_from_training(y, season=168):
    y = np.asarray(y, dtype=float)
    if len(y) <= season:
        return np.nan
    return float(np.mean(np.abs(y[season:] - y[:-season])))

def evaluate_forecast(y_true, y_pred, mase_scale=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        'MAE_MWh': float(mean_absolute_error(y_true, y_pred)),
        'RMSE_MWh': rmse(y_true, y_pred),
        'MAPE_pct': mape(y_true, y_pred),
        'sMAPE_pct': smape(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred)),
    }
    if mase_scale is not None and np.isfinite(mase_scale) and mase_scale > 0:
        out['MASE'] = float(np.mean(np.abs(y_true-y_pred))/mase_scale)
    q = np.quantile(y_true, 0.90)
    mask = y_true >= q
    out['Peak_MAE_top10_MWh'] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return out

def month_to_season_name(m):
    return {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}[int(m)]

def model_size_mb(model):
    try:
        return len(pickle.dumps(model)) / (1024**2)
    except Exception:
        return np.nan

def inference_latency(model, X, repeats=50):
    single = X.iloc[[0]]
    batch = X.iloc[:min(1000, len(X))]
    _ = model.predict(single); _ = model.predict(batch)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(single)
    single_ms = (time.perf_counter()-t0)/repeats*1000
    t0 = time.perf_counter(); _ = model.predict(batch)
    batch_ms_per_row = (time.perf_counter()-t0)/len(batch)*1000
    return float(single_ms), float(batch_ms_per_row)

def save_csv(df, name):
    if SAVE_INTERMEDIATE_CSV and isinstance(df, pd.DataFrame):
        df.to_csv(OUTPUT_DIR / f'{name}.csv', index=False)

def safe_json(obj):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.ndarray,)): return x.tolist()
        if isinstance(x, (pd.Timestamp, datetime)): return str(x)
        return str(x)
    return json.dumps(obj, default=conv, sort_keys=True)

print('✅ Utility functions ready')
