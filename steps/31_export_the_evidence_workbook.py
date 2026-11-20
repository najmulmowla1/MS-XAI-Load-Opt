# Step 31: Export the evidence workbook

print_section('MASTER EXCEL EXPORT')

# Combine final summary.
final_summary_dict={
    'framework':'MS-XAI-Load-Opt',
    'forecast_definition':'At forecast origin t, predict electricity consumption at t+1',
    'selected_model':selected_name,
    'selected_family':SELECTED_FAMILY,
    'selected_optimizer':selected_info['optimizer'],
    'selected_mean_CV_RMSE_MWh':float(selected_cv_df['RMSE_MWh'].mean()),
    'selected_std_CV_RMSE_MWh':float(selected_cv_df['RMSE_MWh'].std(ddof=1)),
    'independent_test_year':TEST_YEAR,
    'main_feature_count':len(FULL_FEATURES),
    'conservative_feature_count':len(FULL_FEATURES_CONSERVATIVE),
    'retained_OWID_indicator_count':len(OWID_FEATURES),
    **{f'test_{k}':v for k,v in final_test_metrics.items()},
}
final_summary_df=pd.DataFrame(list(final_summary_dict.items()),columns=['item','value'])
with open(SUMMARY_JSON_PATH,'w') as f: json.dump(final_summary_dict,f,indent=2,default=str)

run_config_df=pd.DataFrame(list(RUN_CONFIG.items()),columns=['parameter','value'])

# OOF table for the selected model, and full prediction evidence.
selected_oof_export=selected_oof_df.copy()
full_predictions_export=test_results.copy()

# Correlation matrix as tabular sheet.
corr_export=feature_correlation_df.reset_index().rename(columns={'index':'feature'})


# Wide final-test predictions for every benchmark and confirmatory ablation.
all_test_predictions_df=model_df.loc[TEST_IDX,['target_timestamp',TARGET]].copy().reset_index(drop=True)
all_test_predictions_df=all_test_predictions_df.rename(columns={TARGET:'y_true'})
for name,pred in benchmark_predictions.items():
    arr=np.asarray(pred).reshape(-1)
    if len(arr)==len(all_test_predictions_df): all_test_predictions_df[f'pred__{name}']=arr
for name,pred in ablation_test_predictions.items():
    arr=np.asarray(pred).reshape(-1)
    if len(arr)==len(all_test_predictions_df): all_test_predictions_df[f'ablation_pred__{name}']=arr

# Long OOF predictions for default models/baselines, useful for additional pre-test analyses without refitting.
all_cv_oof_rows=[]
for name,frame in cv_oof_predictions.items():
    tmp=frame.copy(); tmp.insert(0,'model',name); all_cv_oof_rows.append(tmp)
all_cv_oof_predictions_df=pd.concat(all_cv_oof_rows,ignore_index=True) if all_cv_oof_rows else pd.DataFrame()

# All conformal interval variants on the 2023 test year.
conformal_intervals_df=model_df.loc[TEST_IDX,['target_timestamp',TARGET]].copy().reset_index(drop=True).rename(columns={TARGET:'y_true'})
conformal_intervals_df['prediction']=final_test_pred
for (method,level),obj in conformal_interval_store.items():
    tag=f'{method}_{int(level*100)}'
    conformal_intervals_df[f'lower__{tag}']=obj['lower']
    conformal_intervals_df[f'upper__{tag}']=obj['upper']

# Sheet registry. Empty frames are skipped.
sheets={
    'evidence_index':evidence_index_df,
    'final_summary':final_summary_df,
    'run_config':run_config_df,
    'data_fingerprints':data_fingerprints_df,
    'package_versions':package_versions_df,
    'hardware':hardware_df,
    'hourly_audit':hourly_audit_df,
    'timezone_audit':timezone_audit_df,
    'holiday_calendar':holiday_calendar_df,
    'price_audit':price_audit_df,
    'price_special_records':price_special_records_df,
    'owid_policy':owid_policy_df,
    'owid_inventory':owid_inventory_df,
    'owid_required_values':owid_required_values_df,
    'horizon_audit':horizon_audit_df,
    'feature_inventory':feature_inventory_df,
    'feature_group_counts':feature_group_counts_df,
    'correlation_matrix':corr_export,
    'top_correlations':top_correlations_df.head(1000),
    'redundancy_summary':redundancy_summary_df,
    'yearly_target_summary':yearly_target_summary_df,
    'cv_folds':cv_folds_df,
    'split_summary':split_summary_df,
    'CV_default_fold':cv_results_df,
    'CV_default_summary':cv_summary_df,
    'ablation_feature_sets':ablation_feature_sets_df,
    'ablation_CV_fold':ablation_cv_df,
    'ablation_CV_summary':ablation_cv_summary_df,
    'search_spaces':search_spaces_df,
    'optimizer_trials':optimizer_trials_df,
    'optimizer_candidates':optimization_candidates_df,
    'selected_CV':selected_cv_df,
    'selected_CV_summary':selected_cv_summary_df,
    'selected_OOF_predictions':selected_oof_export,
    'benchmark_results':benchmark_results_df,
    'prophet_results':prophet_results_df,
    'prophet_errors':prophet_errors_df,
    'dl_config':dl_config_df,
    'dl_results':dl_results_df,
    'dl_errors':dl_errors_df,
    'ablation_test':ablation_test_df,
    'metric_CIs':metric_ci_df,
    'statistical_tests':stat_tests_df,
    'wilcoxon_tests':wilcoxon_df,
    'conformal_summary':conformal_summary_df,
    'conformal_monthly':conformal_monthly_df,
    'SHAP_features':shap_feature_importance_df,
    'SHAP_groups':shap_group_importance_df,
    'SHAP_values_sample':shap_values_sample_df,
    'SHAP_errors':shap_errors_df,
    'error_by_hour':error_by_hour_df,
    'error_by_month':error_by_month_df,
    'error_by_season':error_by_season_df,
    'error_by_weekend':error_by_weekend_df,
    'error_by_weekday':error_by_weekday_df,
    'error_by_holiday':error_by_holiday_df,
    'error_by_peak':error_by_peak_df,
    'error_by_decile':error_by_decile_df,
    'extreme_demand':extreme_demand_df,
    'hard_weeks':hard_weeks_df,
    'operational_profile':operational_profile_df,
    'test_predictions':full_predictions_export,
    'all_test_predictions':all_test_predictions_df,
    'all_CV_OOF_predictions':all_cv_oof_predictions_df,
    'conformal_intervals':conformal_intervals_df,
    'figure_index':figure_index_df,
}

# Excel sheet names must be <=31 characters and unique.
def excel_safe_name(name,used):
    base=re.sub(r'[\\/*?:\[\]]','_',name)[:31]
    candidate=base; i=1
    while candidate in used:
        suffix=f'_{i}'; candidate=base[:31-len(suffix)]+suffix; i+=1
    used.add(candidate); return candidate

used=set(); name_map={k:excel_safe_name(k,used) for k in sheets}

with pd.ExcelWriter(MASTER_EXCEL_PATH,engine='xlsxwriter',datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    workbook=writer.book
    header_fmt=workbook.add_format({'bold':True,'font_color':'white','bg_color':'#1F4E78','border':1,'align':'center','valign':'vcenter','text_wrap':True})
    note_fmt=workbook.add_format({'bg_color':'#FFF2CC','text_wrap':True,'valign':'top'})
    num_fmt=workbook.add_format({'num_format':'0.000'})
    for key,frame in sheets.items():
        if frame is None or not isinstance(frame,pd.DataFrame) or frame.empty:
            continue
        sn=name_map[key]
        export=frame.copy()
        # Excel cannot store tz-aware datetimes. Convert to strings if encountered.
        for c in export.columns:
            if pd.api.types.is_datetime64tz_dtype(export[c]): export[c]=export[c].astype(str)
        export.to_excel(writer,sheet_name=sn,index=False)
        ws=writer.sheets[sn]
        ws.freeze_panes(1,0); ws.autofilter(0,0,len(export),max(len(export.columns)-1,0))
        ws.set_row(0,30,header_fmt)
        for j,c in enumerate(export.columns):
            sample=export[c].astype(str).head(300)
            width=min(max(len(str(c))+2, sample.map(len).max()+2 if len(sample) else 10), 42)
            ws.set_column(j,j,width)
        if key=='evidence_index':
            ws.set_column(0,0,46); ws.set_column(1,1,80)

print('✅ MASTER workbook:',MASTER_EXCEL_PATH)
print('Sheets written:',sum(1 for f in sheets.values() if isinstance(f,pd.DataFrame) and not f.empty))
