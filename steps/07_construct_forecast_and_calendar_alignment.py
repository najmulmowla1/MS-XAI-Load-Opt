# Step 07: Construct forecast and calendar alignment

print_section('FORECAST HORIZON AND CALENDAR ALIGNMENT')

df = hourly.copy()
df['target_timestamp'] = df[TIME_COL]
df['forecast_origin'] = df['target_timestamp'] - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)

# Target-hour calendar variables are known at forecast origin.
df['year'] = df['target_timestamp'].dt.year
df['month'] = df['target_timestamp'].dt.month
df['day'] = df['target_timestamp'].dt.day
df['hour'] = df['target_timestamp'].dt.hour
df['dayofweek'] = df['target_timestamp'].dt.dayofweek
df['dayofyear'] = df['target_timestamp'].dt.dayofyear
df['weekofyear'] = df['target_timestamp'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['dayofweek']>=5).astype(int)
df['season_code'] = df['month'].map(lambda m: {'Winter':0,'Spring':1,'Summer':2,'Autumn':3}[month_to_season_name(m)])

df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)
df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

tr_holidays = holidays.country_holidays('TR', years=range(int(df['year'].min()), int(df['year'].max())+1))
df['is_holiday'] = df['target_timestamp'].dt.date.map(lambda d: int(d in tr_holidays))
df['holiday_name'] = df['target_timestamp'].dt.date.map(lambda d: tr_holidays.get(d, ''))

holiday_calendar_df = pd.DataFrame([
    {'date':str(d),'holiday_name':name,'source':'python-holidays TR calendar'}
    for d,name in sorted(tr_holidays.items())
])

# OWID lag-1 merge
owid1 = owid_tr[['year']+OWID_FEATURES].copy()
owid1['target_year'] = owid1['year'].astype(int) + 1
owid1 = owid1.rename(columns={c:f'owid_{c}_lag1y' for c in OWID_FEATURES})
df = df.merge(owid1.drop(columns=['year']), how='left', left_on='year', right_on='target_year')
df = df.drop(columns=['target_year'])

# OWID lag-2 publication-lag sensitivity
owid2 = owid_tr[['year']+OWID_FEATURES].copy()
owid2['target_year'] = owid2['year'].astype(int) + 2
owid2 = owid2.rename(columns={c:f'owid_{c}_lag2y' for c in OWID_FEATURES})
df = df.merge(owid2.drop(columns=['year']), how='left', left_on='year', right_on='target_year')
df = df.drop(columns=['target_year'])

OWID_LAG1_COLS = [f'owid_{c}_lag1y' for c in OWID_FEATURES]
OWID_LAG2_COLS = [f'owid_{c}_lag2y' for c in OWID_FEATURES]

horizon_audit_df = pd.DataFrame([
    {'element':'forecast origin','notation':'t','dataset_time_relation':'target_timestamp - 1 hour','availability':'forecast issuance time'},
    {'element':'target','notation':'y_(t+1)','dataset_time_relation':'consumption at target_timestamp','availability':'unknown at forecast origin'},
    {'element':'calendar of target hour','notation':'calendar_(t+1)','dataset_time_relation':'derived from target_timestamp','availability':'known in advance'},
    {'element':'demand lag 1','notation':'y_t','dataset_time_relation':'target row shift(1)','availability':'assumed available at forecast origin'},
    {'element':'generation/price lag 1','notation':'g_t / p_t','dataset_time_relation':'target row shift(1)','availability':'main analysis assumes available at forecast origin; conservative sensitivity excludes these'},
    {'element':'OWID lag 1 year','notation':'z_(Y-1)','dataset_time_relation':'target year Y uses OWID year Y-1','availability':'annual context; publication-lag sensitivity uses Y-2'},
])

display(horizon_audit_df)
print('Example alignment:')
display(df[['forecast_origin','target_timestamp',TARGET]].head(5))
