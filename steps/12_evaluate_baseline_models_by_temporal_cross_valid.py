# Step 12: Evaluate baseline models by temporal cross-validation

print_section('ROLLING-ORIGIN CV — BASELINES AND DEFAULT TABULAR MODELS')

cv_results=[]
cv_oof_predictions={}

# Baselines need no fit.
for baseline_name, col in [('Persistence_lag1',f'{TARGET}_lag_1'),('DailyNaive_lag24',f'{TARGET}_lag_24'),('SeasonalNaive_lag168',f'{TARGET}_lag_168')]:
    oof=[]
    for fold in CV_FOLDS:
        tr,va=fold['train_idx'],fold['val_idx']
        ytr=model_df.loc[tr,TARGET]; yva=model_df.loc[va,TARGET]; pred=model_df.loc[va,col].values
        metrics=evaluate_forecast(yva,pred,mase_scale_from_training(ytr.values))
        cv_results.append({'model':baseline_name,'family':'naive_baseline','fold':fold['fold'],'val_year':fold['val_year'],
                           'train_RMSE_MWh':np.nan,'generalization_gap_RMSE_MWh':np.nan,'fit_seconds':0.0,**metrics})
        oof.append(pd.DataFrame({'row_index':va,'target_timestamp':model_df.loc[va,'target_timestamp'].values,'y_true':yva.values,'y_pred':pred,'fold':fold['fold']}))
    cv_oof_predictions[baseline_name]=pd.concat(oof,ignore_index=True)

for name,factory in DEFAULT_MODEL_FACTORIES.items():
    print('CV:',name)
    res,oof=cv_evaluate_factory(factory,FULL_FEATURES,name,return_oof=True)
    res['family']='tabular_default'
    cv_results.extend(res.to_dict('records'))
    cv_oof_predictions[name]=oof

cv_results_df=pd.DataFrame(cv_results)
cv_summary_df=(cv_results_df.groupby(['model','family'],as_index=False)
               .agg(mean_CV_RMSE_MWh=('RMSE_MWh','mean'),std_CV_RMSE_MWh=('RMSE_MWh','std'),
                    mean_CV_MAE_MWh=('MAE_MWh','mean'),mean_CV_MAPE_pct=('MAPE_pct','mean'),
                    mean_generalization_gap_RMSE_MWh=('generalization_gap_RMSE_MWh','mean'),
                    total_fit_seconds=('fit_seconds','sum'))
               .sort_values('mean_CV_RMSE_MWh'))

display(cv_summary_df)
save_csv(cv_results_df,'cv_default_models_fold_results')
save_csv(cv_summary_df,'cv_default_models_summary')
