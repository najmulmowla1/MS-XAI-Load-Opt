# Step 15: Optimize tree models with Optuna TPE

print_section('OPTIMIZATION — OPTUNA/TPE')

optuna.logging.set_verbosity(optuna.logging.WARNING)

if RUN_OPTUNA_XGB:
    def objective_xgb(trial):
        p={
            'learning_rate':trial.suggest_float('learning_rate',0.01,0.15,log=True),
            'max_depth':trial.suggest_int('max_depth',3,10),
            'min_child_weight':trial.suggest_float('min_child_weight',1,20,log=True),
            'subsample':trial.suggest_float('subsample',0.60,1.00),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.60,1.00),
            'reg_alpha':trial.suggest_float('reg_alpha',1e-4,10,log=True),
            'reg_lambda':trial.suggest_float('reg_lambda',1e-3,30,log=True),
            'gamma':trial.suggest_float('gamma',0,5),
        }
        mean_rmse,std_rmse=cv_score_xgb_params(p)
        trial.set_user_attr('std_CV_RMSE_MWh',std_rmse)
        return mean_rmse
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study=optuna.create_study(direction='minimize',sampler=sampler,study_name='xgb_tpe_hourly_load')
    study.optimize(objective_xgb,n_trials=OPTUNA_XGB_TRIALS,show_progress_bar=True)
    bestp=study.best_params
    optimized_candidates['XGBoost_OptunaTPE']={'model_family':'XGBoost','optimizer':'OptunaTPE','params':bestp,
        'mean_CV_RMSE_MWh':float(study.best_value),'std_CV_RMSE_MWh':float(study.best_trial.user_attrs.get('std_CV_RMSE_MWh',np.nan))}
    for t in study.trials:
        optimizer_trials.append({'optimizer':'OptunaTPE_XGBoost','trial':t.number+1,'mean_CV_RMSE_MWh':t.value,
                                 'std_CV_RMSE_MWh':t.user_attrs.get('std_CV_RMSE_MWh',np.nan),'params_json':safe_json(t.params)})

if RUN_OPTUNA_LGBM:
    def objective_lgbm(trial):
        p={
            'learning_rate':trial.suggest_float('learning_rate',0.01,0.15,log=True),
            'num_leaves':trial.suggest_int('num_leaves',15,127),
            'max_depth':trial.suggest_int('max_depth',4,12),
            'min_child_samples':trial.suggest_int('min_child_samples',10,100),
            'subsample':trial.suggest_float('subsample',0.60,1.00),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.60,1.00),
            'reg_alpha':trial.suggest_float('reg_alpha',1e-4,10,log=True),
            'reg_lambda':trial.suggest_float('reg_lambda',1e-3,30,log=True),
        }
        mean_rmse,std_rmse=cv_score_lgbm_params(p)
        trial.set_user_attr('std_CV_RMSE_MWh',std_rmse)
        return mean_rmse
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE+1)
    study_l=optuna.create_study(direction='minimize',sampler=sampler,study_name='lgbm_tpe_hourly_load')
    study_l.optimize(objective_lgbm,n_trials=OPTUNA_LGBM_TRIALS,show_progress_bar=True)
    bestp=study_l.best_params
    optimized_candidates['LightGBM_OptunaTPE']={'model_family':'LightGBM','optimizer':'OptunaTPE','params':bestp,
        'mean_CV_RMSE_MWh':float(study_l.best_value),'std_CV_RMSE_MWh':float(study_l.best_trial.user_attrs.get('std_CV_RMSE_MWh',np.nan))}
    for t in study_l.trials:
        optimizer_trials.append({'optimizer':'OptunaTPE_LightGBM','trial':t.number+1,'mean_CV_RMSE_MWh':t.value,
                                 'std_CV_RMSE_MWh':t.user_attrs.get('std_CV_RMSE_MWh',np.nan),'params_json':safe_json(t.params)})

optimizer_trials_df=pd.DataFrame(optimizer_trials)
save_csv(optimizer_trials_df,'optimizer_trials_after_optuna')
print('Candidates after Optuna:')
for k,v in optimized_candidates.items(): print(k,v['mean_CV_RMSE_MWh'])
