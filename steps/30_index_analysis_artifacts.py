# Step 30: Index analysis artifacts

evidence_index_df=pd.DataFrame([
    {'analysis_topic':'Forecast origin and predictor availability','supporting_outputs':'horizon_audit; availability timing figure; feature_inventory'},
    {'analysis_topic':'Persistence benchmark','supporting_outputs':'benchmark_results; CV_default_summary; statistical_tests'},
    {'analysis_topic':'Expanding-window temporal validation','supporting_outputs':'cv_folds; CV_default_fold; selected_CV; ablation_CV; rolling_origin_cv_design figure'},
    {'analysis_topic':'Feature dependence and redundancy','supporting_outputs':'correlation_matrix; top_correlations; redundancy_summary; generalization gaps in CV'},
    {'analysis_topic':'Model-family comparisons','supporting_outputs':'benchmark_results; prophet_results; dl_results; dl_config'},
    {'analysis_topic':'Full feature inventory','supporting_outputs':'feature_inventory; feature_group_counts; ablation_feature_sets'},
    {'analysis_topic':'Time-zone and holiday conventions','supporting_outputs':'timezone_audit; holiday_calendar; hourly_audit'},
    {'analysis_topic':'Annual indicator preprocessing and publication lag','supporting_outputs':'owid_inventory; owid_policy; E5 vs E8 sensitivity'},
    {'analysis_topic':'Optimization search specifications','supporting_outputs':'search_spaces; optimizer_trials; optimizer_candidates; run_config'},
    {'analysis_topic':'Dependence-aware statistical testing','supporting_outputs':'statistical_tests with Newey-West/Bartlett HAC at 24h and 168h; Holm-adjusted p-values'},
    {'analysis_topic':'Metric confidence intervals','supporting_outputs':'metric_CIs block bootstrap'},
    {'analysis_topic':'Conformal interval evaluation','supporting_outputs':'conformal_summary; conformal_monthly; conformal calibration curve; block-max sensitivity'},
    {'analysis_topic':'Forecast-error stratification','supporting_outputs':'error_by_hour/month/season/weekday/weekend/holiday/peak/decile; hard_weeks'},
    {'analysis_topic':'SHAP interpretation of selected model','supporting_outputs':'SHAP_features; SHAP_groups; selected model name recorded'},
    {'analysis_topic':'Computational assessment','supporting_outputs':'operational_profile; model artifact size; latency/throughput'},
    {'analysis_topic':'Market price audit','supporting_outputs':'price_audit; price_special_records'},
    {'analysis_topic':'Hourly and monthly visualization definitions','supporting_outputs':'separate mean-hourly-by-month figure and monthly-total-GWh figure'},
])
display(evidence_index_df)
