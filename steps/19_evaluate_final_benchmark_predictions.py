# Step 19: Evaluate final benchmark predictions

print_section('FINAL BENCHMARK SUITE — 2023')

benchmark_rows=[]
benchmark_predictions={}
benchmark_models={selected_name:final_model}

# Naive baselines.
for name,col in [('Persistence_lag1',f'{TARGET}_lag_1'),('DailyNaive_lag24',f'{TARGET}_lag_24'),('SeasonalNaive_lag168',f'{TARGET}_lag_168')]:
    pred=model_df.loc[TEST_IDX,col].values
    met=evaluate_forecast(y_test,pred,final_mase_scale)
    benchmark_rows.append({'model':name,'family':'naive_baseline','validation_protocol':'rolling-origin CV available','train_seconds':0.0,**met})
    benchmark_predictions[name]=pred

# Default tabular models.
for name,factory in DEFAULT_MODEL_FACTORIES.items():
    print('Final fit:',name)
    m=factory(); t0=time.perf_counter(); m.fit(X_pre,y_pre); fit_s=time.perf_counter()-t0
    pred=m.predict(X_test); met=evaluate_forecast(y_test,pred,final_mase_scale)
    benchmark_rows.append({'model':name,'family':'tabular_default','validation_protocol':'rolling-origin CV + frozen 2023 test','train_seconds':fit_s,**met})
    benchmark_predictions[name]=pred; benchmark_models[name]=m

# Every optimized candidate, fitted on all pre-test data with its pre-2023 selected parameters.
for name,info in optimized_candidates.items():
    if name==selected_name:
        pred=final_test_pred; fit_s=final_train_seconds; m=final_model
    else:
        if info['model_family']=='XGBoost': m=make_xgb({**info['params'],'n_estimators':N_ESTIMATORS_OPT})
        else: m=make_lgbm({**info['params'],'n_estimators':N_ESTIMATORS_OPT})
        t0=time.perf_counter(); m.fit(X_pre,y_pre); fit_s=time.perf_counter()-t0; pred=m.predict(X_test)
    met=evaluate_forecast(y_test,pred,final_mase_scale)
    benchmark_rows.append({'model':name,'family':'optimizer_enhanced_tree','validation_protocol':'optimizer selected by rolling-origin CV; frozen 2023 test','train_seconds':fit_s,**met})
    benchmark_predictions[name]=pred; benchmark_models[name]=m

benchmark_results_df=pd.DataFrame(benchmark_rows).sort_values('RMSE_MWh').reset_index(drop=True)
display(benchmark_results_df)
save_csv(benchmark_results_df,'benchmark_results_before_prophet_dl')
