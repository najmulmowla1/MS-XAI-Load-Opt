# Step 28: Stratify forecast errors by operational conditions

print_section('ERROR STRATIFICATION AND STRESS DIAGNOSTICS')

test_results=model_df.loc[TEST_IDX,['forecast_origin','target_timestamp',TARGET,'month','hour','dayofweek','is_weekend','is_holiday']].copy().reset_index(drop=True)
test_results['prediction']=final_test_pred
test_results['error_MWh']=test_results[TARGET]-test_results['prediction']
test_results['abs_error_MWh']=np.abs(test_results['error_MWh'])
test_results['squared_error']=test_results['error_MWh']**2
test_results['season']=test_results['month'].map(month_to_season_name)
test_results['peak_top10']=(test_results[TARGET]>=test_results[TARGET].quantile(0.90)).astype(int)
test_results['extreme_top1']=(test_results[TARGET]>=test_results[TARGET].quantile(0.99)).astype(int)
test_results['demand_decile']=pd.qcut(test_results[TARGET],10,labels=False,duplicates='drop')+1
test_results['lower_90']=lo90; test_results['upper_90']=hi90; test_results['covered_90']=conf_detail['covered_90'].values

def grouped_errors(group_col):
    rows=[]
    for key,g in test_results.groupby(group_col,dropna=False):
        met=evaluate_forecast(g[TARGET],g['prediction'],final_mase_scale)
        rows.append({group_col:key,'n':len(g),**met})
    return pd.DataFrame(rows)

error_by_hour_df=grouped_errors('hour')
error_by_month_df=grouped_errors('month')
error_by_season_df=grouped_errors('season')
error_by_weekend_df=grouped_errors('is_weekend')
error_by_weekday_df=grouped_errors('dayofweek')
error_by_holiday_df=grouped_errors('is_holiday')
error_by_peak_df=grouped_errors('peak_top10')
error_by_decile_df=grouped_errors('demand_decile')
extreme_demand_df=grouped_errors('extreme_top1')

# Hard-week detection: rolling 168h MAE, select three non-overlapping highest-error weeks.
roll_mae=test_results['abs_error_MWh'].rolling(168,min_periods=168).mean()
candidates=roll_mae.sort_values(ascending=False).index.tolist()
selected_ends=[]
for idx in candidates:
    if all(abs(idx-j)>=168 for j in selected_ends):
        selected_ends.append(idx)
    if len(selected_ends)==3: break
hard_rows=[]
for rank,end_idx in enumerate(selected_ends,1):
    start_idx=end_idx-167; g=test_results.iloc[start_idx:end_idx+1]
    hard_rows.append({'rank':rank,'start':g['target_timestamp'].min(),'end':g['target_timestamp'].max(),
                      'week_MAE_MWh':g['abs_error_MWh'].mean(),'week_RMSE_MWh':rmse(g[TARGET],g['prediction']),
                      'max_demand_MWh':g[TARGET].max()})
    plt.figure(figsize=(15,5)); plt.plot(g['target_timestamp'],g[TARGET],label='Actual'); plt.plot(g['target_timestamp'],g['prediction'],label='Predicted')
    plt.fill_between(g['target_timestamp'],g['lower_90'],g['upper_90'],alpha=0.2,label='90% conformal interval')
    plt.ylabel('Electricity consumption (MWh)'); plt.xlabel('Target hour'); plt.title(f'Hard Week {rank}: Actual, Forecast and 90% Interval'); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR/f'hard_week_{rank}.png',dpi=300,bbox_inches='tight'); plt.show()
hard_weeks_df=pd.DataFrame(hard_rows)

display(error_by_hour_df.head())
display(hard_weeks_df)
