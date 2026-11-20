# Step 24: Estimate uncertainty of evaluation metrics

print_section('BLOCK-BOOTSTRAP CONFIDENCE INTERVALS')

def circular_block_indices(n,block_size,rng):
    n_blocks=int(np.ceil(n/block_size))
    starts=rng.integers(0,n,size=n_blocks)
    idx=[]
    for s in starts:
        idx.extend((s+np.arange(block_size))%n)
    return np.asarray(idx[:n],dtype=int)

def block_bootstrap_metric_ci(y_true,y_pred,reps=BOOTSTRAP_REPS,block_size=BOOTSTRAP_BLOCK_HOURS,ci=BOOTSTRAP_CI,seed=RANDOM_STATE):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); rng=np.random.default_rng(seed)
    values={'MAE_MWh':[],'RMSE_MWh':[]}
    for _ in range(reps):
        idx=circular_block_indices(len(y_true),block_size,rng)
        values['MAE_MWh'].append(mean_absolute_error(y_true[idx],y_pred[idx]))
        values['RMSE_MWh'].append(rmse(y_true[idx],y_pred[idx]))
    alpha=1-ci; rows=[]
    for metric,arr in values.items():
        arr=np.asarray(arr); point=mean_absolute_error(y_true,y_pred) if metric=='MAE_MWh' else rmse(y_true,y_pred)
        rows.append({'metric':metric,'point_estimate':point,'CI_level':ci,'lower':np.quantile(arr,alpha/2),'upper':np.quantile(arr,1-alpha/2),
                     'bootstrap_reps':reps,'block_hours':block_size})
    return pd.DataFrame(rows)

# Key models for inferential reporting.
key_ci_models=[selected_name,'Persistence_lag1','SeasonalNaive_lag168','XGBoost_default']
# Add strongest non-proposed benchmark by test RMSE as descriptive sensitivity, without using it for model selection.
others=benchmark_results_df[~benchmark_results_df['model'].isin(key_ci_models+[selected_name])]
if not others.empty: key_ci_models.append(others.iloc[0]['model'])

ci_rows=[]
for i,name in enumerate(dict.fromkeys(key_ci_models)):
    if name in benchmark_predictions:
        ci=block_bootstrap_metric_ci(y_test.values,benchmark_predictions[name],seed=RANDOM_STATE+i)
        ci.insert(0,'model',name); ci_rows.append(ci)
metric_ci_df=pd.concat(ci_rows,ignore_index=True) if ci_rows else pd.DataFrame()
display(metric_ci_df)
