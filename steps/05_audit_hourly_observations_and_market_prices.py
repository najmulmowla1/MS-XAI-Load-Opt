# Step 05: Audit hourly observations and market prices

print_section('HOURLY DATA / TIMEZONE / PRICE AUDIT')

time_candidates = [c for c in hourly.columns if c.lower() in ['time','timestamp','date','datetime']]
TIME_COL = time_candidates[0] if time_candidates else hourly.columns[0]
hourly[TIME_COL] = parse_hourly_datetime(hourly[TIME_COL])
hourly = hourly.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

if TARGET not in hourly.columns:
    raise ValueError(f"Target '{TARGET}' not found. Columns: {hourly.columns.tolist()}")
hourly[TARGET] = pd.to_numeric(hourly[TARGET], errors='coerce')

expected = pd.date_range(hourly[TIME_COL].min(), hourly[TIME_COL].max(), freq='h')
missing_hours = expected.difference(hourly[TIME_COL])

timezone_rows=[]
tz = ZoneInfo(ASSUMED_TIMEZONE)
for year in sorted(hourly[TIME_COL].dt.year.unique()):
    for month in [1,4,7,10]:
        local_dt = datetime(int(year), month, 1, 12, 0, tzinfo=tz)
        timezone_rows.append({
            'year':int(year),'month':month,'timezone':ASSUMED_TIMEZONE,
            'UTC_offset_hours':local_dt.utcoffset().total_seconds()/3600,
            'DST_hours':local_dt.dst().total_seconds()/3600 if local_dt.dst() else 0.0
        })
timezone_audit_df = pd.DataFrame(timezone_rows)

hourly_audit_df = pd.DataFrame([
    {'item':'rows','value':len(hourly)},
    {'item':'columns','value':hourly.shape[1]},
    {'item':'time_column','value':TIME_COL},
    {'item':'start_target_timestamp','value':hourly[TIME_COL].min()},
    {'item':'end_target_timestamp','value':hourly[TIME_COL].max()},
    {'item':'missing_target','value':int(hourly[TARGET].isna().sum())},
    {'item':'duplicate_timestamps','value':int(hourly[TIME_COL].duplicated().sum())},
    {'item':'expected_hourly_rows','value':len(expected)},
    {'item':'missing_hourly_timestamps','value':len(missing_hours)},
    {'item':'timezone_assumption','value':ASSUMED_TIMEZONE},
    {'item':'timestamp_interpretation','value':'naive source timestamps interpreted as Türkiye local wall-clock target delivery hours'},
])

price_candidates = ['TRY_MWh','USD_MWh','EUR_MWh']
price_cols = [c for c in price_candidates if c in hourly.columns]
price_audit=[]
for c in price_cols:
    s = pd.to_numeric(hourly[c], errors='coerce')
    price_audit.append({
        'variable':c,'count':int(s.notna().sum()),'missing':int(s.isna().sum()),
        'minimum':float(s.min()),'maximum':float(s.max()),
        'zero_count':int((s==0).sum()),'negative_count':int((s<0).sum()),
        'zero_share_pct':float((s==0).mean()*100),'negative_share_pct':float((s<0).mean()*100)
    })
price_audit_df = pd.DataFrame(price_audit)

if price_cols:
    zero_mask = np.zeros(len(hourly), dtype=bool)
    negative_mask = np.zeros(len(hourly), dtype=bool)
    for c in price_cols:
        s = pd.to_numeric(hourly[c], errors='coerce')
        zero_mask |= (s==0).values
        negative_mask |= (s<0).values
    price_special_records_df = hourly.loc[zero_mask | negative_mask, [TIME_COL]+price_cols].copy()
else:
    price_special_records_df = pd.DataFrame()

display(hourly_audit_df)
display(price_audit_df)
print('Timezone unique UTC offsets:', sorted(timezone_audit_df['UTC_offset_hours'].unique()))

save_csv(hourly_audit_df,'hourly_audit')
save_csv(price_audit_df,'price_audit')
