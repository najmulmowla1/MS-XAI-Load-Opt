# Step 14: Optimize XGBoost with grid and random search

print_section('OPTIMIZATION — GRID + RANDOM SEARCH')

optimizer_trials=[]
optimized_candidates={}

# Exactly 24 predefined grid combinations.
grid_spec={
    'learning_rate':[0.03,0.07,0.12],
    'max_depth':[4,7],
    'min_child_weight':[1.0,5.0],
    'subsample':[0.80,1.00],
}
grid_params=list(ParameterGrid(grid_spec))
assert len(grid_params)==24
for p in grid_params:
    p.update({'colsample_bytree':0.85,'reg_alpha':0.0,'reg_lambda':1.0,'gamma':0.0})

if RUN_GRID_SEARCH:
    best=None
    for i,p in enumerate(grid_params,1):
        mean_rmse,std_rmse=cv_score_xgb_params(p)
        row={'optimizer':'GridSearch_XGBoost','trial':i,'mean_CV_RMSE_MWh':mean_rmse,'std_CV_RMSE_MWh':std_rmse,'params_json':safe_json(p)}
        optimizer_trials.append(row)
        if best is None or mean_rmse<best[0]: best=(mean_rmse,std_rmse,p.copy())
        print(f'Grid {i:02d}/24 RMSE={mean_rmse:.3f}')
    optimized_candidates['XGBoost_GridSearch']={'model_family':'XGBoost','optimizer':'GridSearch','params':best[2],'mean_CV_RMSE_MWh':best[0],'std_CV_RMSE_MWh':best[1]}

rng=np.random.default_rng(RANDOM_STATE)
def sample_xgb_params(rng):
    return {
        'learning_rate':float(np.exp(rng.uniform(np.log(0.01),np.log(0.15)))),
        'max_depth':int(rng.integers(3,11)),
        'min_child_weight':float(np.exp(rng.uniform(np.log(1),np.log(20)))),
        'subsample':float(rng.uniform(0.60,1.00)),
        'colsample_bytree':float(rng.uniform(0.60,1.00)),
        'reg_alpha':float(np.exp(rng.uniform(np.log(1e-4),np.log(10)))),
        'reg_lambda':float(np.exp(rng.uniform(np.log(1e-3),np.log(30)))),
        'gamma':float(rng.uniform(0,5)),
    }

if RUN_RANDOM_SEARCH:
    best=None
    for i in range(1,RANDOM_SEARCH_TRIALS+1):
        p=sample_xgb_params(rng)
        mean_rmse,std_rmse=cv_score_xgb_params(p)
        optimizer_trials.append({'optimizer':'RandomSearch_XGBoost','trial':i,'mean_CV_RMSE_MWh':mean_rmse,'std_CV_RMSE_MWh':std_rmse,'params_json':safe_json(p)})
        if best is None or mean_rmse<best[0]: best=(mean_rmse,std_rmse,p.copy())
        print(f'Random {i:02d}/{RANDOM_SEARCH_TRIALS} RMSE={mean_rmse:.3f}')
    optimized_candidates['XGBoost_RandomSearch']={'model_family':'XGBoost','optimizer':'RandomSearch','params':best[2],'mean_CV_RMSE_MWh':best[0],'std_CV_RMSE_MWh':best[1]}

optimizer_trials_df=pd.DataFrame(optimizer_trials)
save_csv(optimizer_trials_df,'optimizer_trials_partial')
print('Partial candidates:',optimized_candidates)
