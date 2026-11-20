# Step 17: Select a forecasting model using validation only

print_section('VALIDATION-SELECTED PROPOSED MODEL')

if optimization_candidates_df.empty:
    raise RuntimeError('No optimized candidate available.')
selected_name=optimization_candidates_df.iloc[0]['candidate_model']
selected_info=optimized_candidates[selected_name]
SELECTED_PARAMS=selected_info['params']
SELECTED_FAMILY=selected_info['model_family']

if SELECTED_FAMILY=='XGBoost':
    selected_factory=lambda:make_xgb({**SELECTED_PARAMS,'n_estimators':N_ESTIMATORS_OPT})
else:
    selected_factory=lambda:make_lgbm({**SELECTED_PARAMS,'n_estimators':N_ESTIMATORS_OPT})

selected_cv_df,selected_oof_df=cv_evaluate_factory(selected_factory,FULL_FEATURES,selected_name,return_oof=True)
selected_cv_summary_df=pd.DataFrame([{
    'selected_model':selected_name,'model_family':SELECTED_FAMILY,'optimizer':selected_info['optimizer'],
    'mean_CV_RMSE_MWh':selected_cv_df['RMSE_MWh'].mean(),'std_CV_RMSE_MWh':selected_cv_df['RMSE_MWh'].std(ddof=1),
    'mean_CV_MAE_MWh':selected_cv_df['MAE_MWh'].mean(),'mean_CV_MAPE_pct':selected_cv_df['MAPE_pct'].mean(),
    'mean_generalization_gap_RMSE_MWh':selected_cv_df['generalization_gap_RMSE_MWh'].mean(),
    'params_json':safe_json(SELECTED_PARAMS)
}])

display(selected_cv_summary_df)
print('🔒 Selected entirely from pre-2023 rolling-origin CV.')
save_csv(selected_oof_df,'selected_model_oof_predictions')
