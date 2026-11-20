# Step 26: Construct conformal prediction intervals

print_section('CONFORMAL UNCERTAINTY — PRE-TEST OOF CALIBRATION')

def conformal_quantile(abs_residuals,alpha):
    r=np.sort(np.asarray(abs_residuals,float)); n=len(r)
    rank=int(np.ceil((n+1)*(1-alpha))); rank=min(max(rank,1),n)
    return float(r[rank-1])

def block_max_quantile(abs_residuals,alpha,block_size):
    r=np.asarray(abs_residuals,float); n_blocks=len(r)//block_size
    if n_blocks<5: return conformal_quantile(r,alpha)
    block_max=r[:n_blocks*block_size].reshape(n_blocks,block_size).max(axis=1)
    return conformal_quantile(block_max,alpha)

def interval_metrics(y,lower,upper,alpha):
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    covered=(y>=lower)&(y<=upper); width=upper-lower
    wink=width.copy(); below=y<lower; above=y>upper
    wink[below]+=(2/alpha)*(lower[below]-y[below]); wink[above]+=(2/alpha)*(y[above]-upper[above])
    return {'empirical_coverage_pct':float(covered.mean()*100),'mean_width_MWh':float(width.mean()),
            'normalized_width':float(width.mean()/(y.max()-y.min())),'Winkler_score':float(wink.mean())}

cal_resid=np.abs(selected_oof_df['y_true'].values-selected_oof_df['y_pred'].values)
conformal_rows=[]; conformal_interval_store={}
for level in CONFORMAL_NOMINAL_LEVELS:
    alpha=1-level; q=conformal_quantile(cal_resid,alpha); lo=final_test_pred-q; hi=final_test_pred+q
    met=interval_metrics(y_test.values,lo,hi,alpha)
    conformal_rows.append({'method':'simple_split_conformal_OOF','nominal_coverage_pct':level*100,'alpha':alpha,'q_MWh':q,
                           'calibration_source':'pooled rolling-origin OOF residuals, validation years 2020–2022',**met})
    conformal_interval_store[('simple',level)]={'lower':lo,'upper':hi}

# Dependence sensitivity at 90% using block maxima.
for block in CONFORMAL_BLOCK_SENSITIVITY:
    level=0.90; alpha=0.10; q=block_max_quantile(cal_resid,alpha,block); lo=final_test_pred-q; hi=final_test_pred+q
    met=interval_metrics(y_test.values,lo,hi,alpha)
    conformal_rows.append({'method':f'block_max_conformal_{block}h','nominal_coverage_pct':90.0,'alpha':alpha,'q_MWh':q,
                           'calibration_source':'pooled rolling-origin OOF residuals; non-overlapping block maxima',**met})
    conformal_interval_store[(f'block{block}',0.90)]={'lower':lo,'upper':hi}

conformal_summary_df=pd.DataFrame(conformal_rows)

# Month-level coverage for the 90% simple interval.
lo90=conformal_interval_store[('simple',0.90)]['lower']; hi90=conformal_interval_store[('simple',0.90)]['upper']
conf_detail=model_df.loc[TEST_IDX,['target_timestamp',TARGET]].copy().reset_index(drop=True)
conf_detail['prediction']=final_test_pred; conf_detail['lower_90']=lo90; conf_detail['upper_90']=hi90
conf_detail['covered_90']=((conf_detail[TARGET]>=lo90)&(conf_detail[TARGET]<=hi90)).astype(int)
conf_detail['month']=conf_detail['target_timestamp'].dt.month
conformal_monthly_df=conf_detail.groupby('month').agg(coverage_pct=('covered_90',lambda x:100*x.mean()),mean_width_MWh=('lower_90',lambda x:0)).reset_index()
conformal_monthly_df['mean_width_MWh']=[(conf_detail.loc[conf_detail['month']==m,'upper_90']-conf_detail.loc[conf_detail['month']==m,'lower_90']).mean() for m in conformal_monthly_df['month']]

display(conformal_summary_df)
