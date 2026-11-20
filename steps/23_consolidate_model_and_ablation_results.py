# Step 23: Consolidate model and ablation results

print_section('CONSOLIDATED FINAL MODEL RESULTS')

benchmark_results_df=pd.DataFrame(benchmark_rows).drop_duplicates(subset=['model'],keep='last').sort_values('RMSE_MWh').reset_index(drop=True)

# Confirmatory final-test ablation; parameters fixed before inspecting 2023.
ablation_test_rows=[]
ablation_test_predictions={}
for exp,features in ABLATION_SETS.items():
    Xpre=model_df.loc[PRETEST_IDX,features].astype(np.float32); Xte=model_df.loc[TEST_IDX,features].astype(np.float32)
    m=make_xgb(AB_MODEL_PARAMS); m.fit(Xpre,y_pre); pred=m.predict(Xte)
    ablation_test_predictions[exp]=pred
    ablation_test_rows.append({'experiment':exp,'feature_count':len(features),**evaluate_forecast(y_test,pred,final_mase_scale)})
    del m,Xpre,Xte; gc.collect()
ablation_test_df=pd.DataFrame(ablation_test_rows).sort_values('RMSE_MWh').reset_index(drop=True)

display(benchmark_results_df)
display(ablation_test_df)
save_csv(benchmark_results_df,'benchmark_results_final')
save_csv(ablation_test_df,'ablation_test_confirmatory')
