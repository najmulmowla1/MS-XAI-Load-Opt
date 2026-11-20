# Step 18: Fit selected model and assess held-out observations

print_section('VALIDATION-SELECTED MODEL — INDEPENDENT 2023 TEST')

X_pre=model_df.loc[PRETEST_IDX,FULL_FEATURES].astype(np.float32)
y_pre=model_df.loc[PRETEST_IDX,TARGET].astype(float)
X_test=model_df.loc[TEST_IDX,FULL_FEATURES].astype(np.float32)
y_test=model_df.loc[TEST_IDX,TARGET].astype(float)

rss_before=proc.memory_info().rss/1024**2
final_model=selected_factory()
t0=time.perf_counter(); final_model.fit(X_pre,y_pre); final_train_seconds=time.perf_counter()-t0
rss_after=proc.memory_info().rss/1024**2
final_test_pred=final_model.predict(X_test)
final_mase_scale=mase_scale_from_training(y_pre.values)
final_test_metrics=evaluate_forecast(y_test,final_test_pred,final_mase_scale)
single_ms,batch_ms=inference_latency(final_model,X_test)

final_model_path=MODEL_DIR / 'selected_proposed_model.pkl'
joblib.dump(final_model,final_model_path)
serialized_size_mb=final_model_path.stat().st_size/1024**2

operational_profile_df=pd.DataFrame([{
    'selected_model':selected_name,'training_seconds_pretest_2018_2022':final_train_seconds,
    'RSS_before_MB':rss_before,'RSS_after_MB':rss_after,'RSS_delta_MB':rss_after-rss_before,
    'serialized_model_size_MB':serialized_size_mb,'single_row_latency_ms':single_ms,
    'batch_latency_ms_per_row':batch_ms,'batch_throughput_rows_per_second':1000/batch_ms if batch_ms>0 else np.nan,
    'feature_count':len(FULL_FEATURES)
}])
final_test_metrics_df=pd.DataFrame([{'model':selected_name,'family':'proposed_optimized_tree',**final_test_metrics}])

display(final_test_metrics_df)
display(operational_profile_df)
