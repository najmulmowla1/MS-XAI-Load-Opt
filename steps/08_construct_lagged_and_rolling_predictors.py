# Step 08: Construct lagged and rolling predictors

print_section('FEATURE ENGINEERING AND AVAILABILITY INVENTORY')

generation_candidates = [
    'total_generation_MWh','natural_gas','hydro_dam','lignite','hydro_river','coal_imported',
    'wind','solar','fuel_oil','geothermal','asphaltite_coal','hard_coal','biomass','naphtha',
    'LNG','international','waste_heat'
]
price_candidates = ['TRY_MWh','USD_MWh','EUR_MWh']
GEN_COLS = [c for c in generation_candidates if c in df.columns]
PRICE_COLS = [c for c in price_candidates if c in df.columns]

for c in [TARGET]+GEN_COLS+PRICE_COLS+OWID_LAG1_COLS+OWID_LAG2_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce')

TEMPORAL_COLS = ['year','month','day','hour','dayofweek','dayofyear','weekofyear','is_weekend','is_holiday','season_code',
                 'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos']

TARGET_LAG_COLS=[]
for lag in TARGET_LAGS:
    col=f'{TARGET}_lag_{lag}'
    df[col]=df[TARGET].shift(lag)
    TARGET_LAG_COLS.append(col)

ROLLING_COLS=[]
shifted_y = df[TARGET].shift(1)
for w in ROLL_WINDOWS:
    for stat in ['mean','std','min','max']:
        col=f'{TARGET}_roll_{stat}_{w}'
        if stat=='mean': df[col]=shifted_y.rolling(w).mean()
        elif stat=='std': df[col]=shifted_y.rolling(w).std()
        elif stat=='min': df[col]=shifted_y.rolling(w).min()
        else: df[col]=shifted_y.rolling(w).max()
        ROLLING_COLS.append(col)

PRICE_LAG_COLS=[]
GEN_LAG_COLS=[]
for c in PRICE_COLS:
    for lag in OPERATIONAL_LAGS:
        col=f'{c}_lag_{lag}'
        df[col]=df[c].shift(lag)
        PRICE_LAG_COLS.append(col)
for c in GEN_COLS:
    for lag in OPERATIONAL_LAGS:
        col=f'{c}_lag_{lag}'
        df[col]=df[c].shift(lag)
        GEN_LAG_COLS.append(col)

PRICE_CONSERVATIVE_COLS=[c for c in PRICE_LAG_COLS if not c.endswith('_lag_1')]
GEN_CONSERVATIVE_COLS=[c for c in GEN_LAG_COLS if not c.endswith('_lag_1')]

FEATURE_GROUPS = {
    'temporal': TEMPORAL_COLS,
    'target_lag': TARGET_LAG_COLS,
    'rolling': ROLLING_COLS,
    'price_lag': PRICE_LAG_COLS,
    'generation_lag': GEN_LAG_COLS,
    'owid_lag1y': OWID_LAG1_COLS,
    'owid_lag2y': OWID_LAG2_COLS,
}

DEMAND_HISTORY_FEATURES = TEMPORAL_COLS + TARGET_LAG_COLS + ROLLING_COLS
FULL_FEATURES = DEMAND_HISTORY_FEATURES + PRICE_LAG_COLS + GEN_LAG_COLS + OWID_LAG1_COLS
FULL_FEATURES_CONSERVATIVE = DEMAND_HISTORY_FEATURES + PRICE_CONSERVATIVE_COLS + GEN_CONSERVATIVE_COLS + OWID_LAG1_COLS
DEMAND_PLUS_OWID_LAG1 = DEMAND_HISTORY_FEATURES + OWID_LAG1_COLS
DEMAND_PLUS_OWID_LAG2 = DEMAND_HISTORY_FEATURES + OWID_LAG2_COLS

# Feature inventory with auditable timing.
inventory=[]
for group, cols in FEATURE_GROUPS.items():
    for c in cols:
        definition=''
        relative_target='known/calendar'
        relative_origin='known in advance'
        assumption='known at forecast origin'
        if group=='target_lag':
            lag=int(c.rsplit('_',1)[1]); definition=f'consumption {lag} hour(s) before target hour'
            relative_target=f'-{lag}h'; relative_origin=f'{1-lag:+d}h relative to origin'
            assumption='historical demand available by forecast origin'
        elif group=='rolling':
            w=int(c.rsplit('_',1)[1]); definition=f'shifted historical consumption rolling statistic, window={w}h'
            relative_target='ends at -1h'; relative_origin='ends at origin'
            assumption='uses demand no later than forecast origin'
        elif group in ['price_lag','generation_lag']:
            lag=int(c.rsplit('_',1)[1]); definition=f'{group.replace("_lag","")} variable lagged {lag}h relative to target'
            relative_target=f'-{lag}h'; relative_origin=f'{1-lag:+d}h relative to origin'
            assumption='lag-1 availability explicitly tested with conservative sensitivity' if lag==1 else 'historical operational value'
        elif group=='owid_lag1y':
            definition='annual OWID indicator from target year Y-1'; relative_target='-1 year'; relative_origin='annual context'
            assumption='main annual-context assumption; publication-lag sensitivity also evaluated'
        elif group=='owid_lag2y':
            definition='annual OWID indicator from target year Y-2'; relative_target='-2 years'; relative_origin='annual context'
            assumption='conservative publication-lag sensitivity'
        else:
            definition='target-hour calendar/cyclic feature'; relative_target='target hour'; relative_origin='+1h calendar known in advance'
        inventory.append({'feature':c,'group':group,'definition':definition,
                          'relative_to_target':relative_target,'relative_to_forecast_origin':relative_origin,
                          'availability_assumption':assumption,
                          'used_main_full':c in FULL_FEATURES,
                          'used_conservative_full':c in FULL_FEATURES_CONSERVATIVE})
feature_inventory_df = pd.DataFrame(inventory)
feature_group_counts_df = feature_inventory_df.groupby('group').agg(
    total_features=('feature','count'), main_full_features=('used_main_full','sum'), conservative_full_features=('used_conservative_full','sum')
).reset_index()

# Drop rows that cannot support the maximum lag/rolling history or main/sensitivity OWID features.
required_model_cols = list(dict.fromkeys(FULL_FEATURES + FULL_FEATURES_CONSERVATIVE + DEMAND_PLUS_OWID_LAG2 + [TARGET]))
model_df = df.dropna(subset=required_model_cols).copy().reset_index(drop=True)
model_df['target_year'] = model_df['target_timestamp'].dt.year

print('Usable model rows:', len(model_df))
print('Main full feature count:', len(FULL_FEATURES))
print('Conservative feature count:', len(FULL_FEATURES_CONSERVATIVE))
display(feature_group_counts_df)
