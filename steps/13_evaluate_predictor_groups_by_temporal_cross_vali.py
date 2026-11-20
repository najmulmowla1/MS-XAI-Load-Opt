# Step 13: Evaluate predictor groups by temporal cross-validation

print_section('ABLATION — ROLLING-ORIGIN CV')

ABLATION_SETS = {
    'E1_temporal_only':TEMPORAL_COLS,
    'E2_demand_history':DEMAND_HISTORY_FEATURES,
    'E3_E2_plus_price':DEMAND_HISTORY_FEATURES+PRICE_LAG_COLS,
    'E4_E2_plus_generation':DEMAND_HISTORY_FEATURES+GEN_LAG_COLS,
    'E5_E2_plus_OWID_lag1y':DEMAND_PLUS_OWID_LAG1,
    'E6_full_multisource':FULL_FEATURES,
    'E7_full_conservative_operational_latency':FULL_FEATURES_CONSERVATIVE,
    'E8_E2_plus_OWID_lag2y_publication_sensitivity':DEMAND_PLUS_OWID_LAG2,
}

ablation_feature_sets_df=pd.DataFrame([
    {'experiment':k,'feature_count':len(v),'features_json':safe_json(v)} for k,v in ABLATION_SETS.items()
])

ablation_cv_rows=[]
AB_MODEL_PARAMS=dict(n_estimators=500,learning_rate=0.05,max_depth=6,min_child_weight=1,subsample=0.9,
                     colsample_bytree=0.9,reg_alpha=0,reg_lambda=1,gamma=0)
for exp,features in ABLATION_SETS.items():
    print('Ablation CV:',exp,'features=',len(features))
    res=cv_evaluate_factory(lambda p=AB_MODEL_PARAMS:make_xgb(p),features,exp,return_oof=False)
    res['experiment']=exp
    ablation_cv_rows.extend(res.to_dict('records'))

ablation_cv_df=pd.DataFrame(ablation_cv_rows)
ablation_cv_summary_df=(ablation_cv_df.groupby('experiment',as_index=False)
                        .agg(feature_count=('model','size'),mean_CV_RMSE_MWh=('RMSE_MWh','mean'),std_CV_RMSE_MWh=('RMSE_MWh','std'),
                             mean_CV_MAE_MWh=('MAE_MWh','mean'),mean_CV_MAPE_pct=('MAPE_pct','mean'),
                             mean_train_RMSE_MWh=('train_RMSE_MWh','mean'),
                             mean_generalization_gap_RMSE_MWh=('generalization_gap_RMSE_MWh','mean')))
ablation_cv_summary_df=ablation_cv_summary_df.merge(ablation_feature_sets_df[['experiment','feature_count']],on='experiment',suffixes=('','_true'))
ablation_cv_summary_df['feature_count']=ablation_cv_summary_df['feature_count_true']; ablation_cv_summary_df=ablation_cv_summary_df.drop(columns=['feature_count_true'])
ablation_cv_summary_df=ablation_cv_summary_df.sort_values('mean_CV_RMSE_MWh')

display(ablation_cv_summary_df)
save_csv(ablation_cv_df,'ablation_cv_fold_results')
save_csv(ablation_cv_summary_df,'ablation_cv_summary')
