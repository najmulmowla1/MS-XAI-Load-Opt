# Step 20: Evaluate Prophet benchmarks

print_section('PROPHET BENCHMARKS')

prophet_results=[]
prophet_errors=[]
if RUN_PROPHET:
    try:
        from prophet import Prophet
        # Fixed pre-test validation protocol: 2018–2021 train, 2022 validation to audit configuration,
        # then refit 2018–2022 and evaluate 2023 once.
        p_train_mask=model_df['target_year']<=2021
        p_val_mask=model_df['target_year']==2022
        p_test_mask=model_df['target_year']==2023
        reg_cols=[f'{TARGET}_lag_1',f'{TARGET}_lag_24',f'{TARGET}_lag_168','hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','is_weekend','is_holiday']

        def prophet_fit_predict(with_regressors=False):
            train_cols=['target_timestamp',TARGET]+(reg_cols if with_regressors else [])
            train=model_df.loc[p_train_mask,train_cols].copy().rename(columns={'target_timestamp':'ds',TARGET:'y'})
            val=model_df.loc[p_val_mask,train_cols].copy().rename(columns={'target_timestamp':'ds',TARGET:'y'})
            m=Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True,
                      changepoint_prior_scale=0.05,seasonality_mode='additive')
            if with_regressors:
                for c in reg_cols: m.add_regressor(c)
            t0=time.perf_counter(); m.fit(train); tune_s=time.perf_counter()-t0
            val_pred=m.predict(val[['ds']+(reg_cols if with_regressors else [])])['yhat'].values
            val_metrics=evaluate_forecast(val['y'].values,val_pred,mase_scale_from_training(train['y'].values))
            # refit through 2022
            pre=model_df.loc[model_df['target_year']<=2022,train_cols].copy().rename(columns={'target_timestamp':'ds',TARGET:'y'})
            test=model_df.loc[p_test_mask,train_cols].copy().rename(columns={'target_timestamp':'ds',TARGET:'y'})
            m2=Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True,
                       changepoint_prior_scale=0.05,seasonality_mode='additive')
            if with_regressors:
                for c in reg_cols: m2.add_regressor(c)
            t0=time.perf_counter(); m2.fit(pre); final_s=time.perf_counter()-t0
            test_pred=m2.predict(test[['ds']+(reg_cols if with_regressors else [])])['yhat'].values
            test_metrics=evaluate_forecast(test['y'].values,test_pred,final_mase_scale)
            return val_metrics,test_metrics,test_pred,tune_s+final_s

        for name,reg in [('Prophet_univariate',False),('Prophet_with_regressors',True)]:
            vm,tm,pred,fit_s=prophet_fit_predict(reg)
            prophet_results.append({'model':name,'family':'statistical_prophet','validation_protocol':'2018–2021 train / 2022 validation; refit through 2022','train_seconds':fit_s,
                                    **{f'val_{k}':v for k,v in vm.items()},**tm})
            benchmark_predictions[name]=pred
            benchmark_rows.append({'model':name,'family':'statistical_prophet','validation_protocol':'2018–2021 train / 2022 validation; refit through 2022','train_seconds':fit_s,**tm})
    except Exception as e:
        prophet_errors.append({'component':'Prophet','error':repr(e)})
        print('Prophet failed:',repr(e))

prophet_results_df=pd.DataFrame(prophet_results)
prophet_errors_df=pd.DataFrame(prophet_errors)
if not prophet_results_df.empty: display(prophet_results_df)
