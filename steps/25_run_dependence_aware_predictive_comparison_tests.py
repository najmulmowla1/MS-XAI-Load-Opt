# Step 25: Run dependence-aware predictive comparison tests

print_section('STATISTICAL TESTS — HAC DM + HOLM')

def dm_hac(errors_a,errors_b,power=2,hac_lags=24):
    ea=np.asarray(errors_a,float); eb=np.asarray(errors_b,float)
    d=(ea**2-eb**2) if power==2 else (np.abs(ea)-np.abs(eb))
    d=d[np.isfinite(d)]; n=len(d); dbar=d.mean(); z=d-dbar
    gamma0=np.dot(z,z)/n; lags=min(hac_lags,n-1); longvar=gamma0
    for k in range(1,lags+1):
        gamma=np.dot(z[k:],z[:-k])/n
        bartlett=1-k/(lags+1)
        longvar += 2*bartlett*gamma
    if longvar<=0: return {'DM_stat':np.nan,'p_two_sided':np.nan,'mean_loss_diff':dbar,'n':n,'HAC_lags':lags}
    stat=dbar/np.sqrt(longvar/n); p=2*(1-norm.cdf(abs(stat)))
    return {'DM_stat':float(stat),'p_two_sided':float(p),'mean_loss_diff':float(dbar),'n':int(n),'HAC_lags':int(lags)}

comparisons=[]
def add_dm(label,pred_a,pred_b,primary=True):
    ea=y_test.values-np.asarray(pred_a); eb=y_test.values-np.asarray(pred_b)
    for lag in HAC_LAGS_SENSITIVITY:
        r=dm_hac(ea,eb,power=2,hac_lags=lag)
        r.update({'comparison':label,'model_A':label.split(' vs ')[0],'model_B':label.split(' vs ')[1] if ' vs ' in label else '',
                  'loss':'squared_error','primary_HAC_specification':lag==HAC_LAGS_PRIMARY,'interpretation_sign':'negative DM => model A lower loss'})
        comparisons.append(r)

for b in ['Persistence_lag1','DailyNaive_lag24','SeasonalNaive_lag168','XGBoost_default']:
    if b in benchmark_predictions: add_dm(f'{selected_name} vs {b}',final_test_pred,benchmark_predictions[b])
if 'E2_demand_history' in ablation_test_predictions and 'E5_E2_plus_OWID_lag1y' in ablation_test_predictions:
    add_dm('E5_OWID_lag1 vs E2_demand_history',ablation_test_predictions['E5_E2_plus_OWID_lag1y'],ablation_test_predictions['E2_demand_history'])
if 'E6_full_multisource' in ablation_test_predictions and 'E2_demand_history' in ablation_test_predictions:
    add_dm('E6_full_multisource vs E2_demand_history',ablation_test_predictions['E6_full_multisource'],ablation_test_predictions['E2_demand_history'])
if 'E7_full_conservative_operational_latency' in ablation_test_predictions and 'E6_full_multisource' in ablation_test_predictions:
    add_dm('E7_conservative_latency vs E6_full_multisource',ablation_test_predictions['E7_full_conservative_operational_latency'],ablation_test_predictions['E6_full_multisource'])
if 'E8_E2_plus_OWID_lag2y_publication_sensitivity' in ablation_test_predictions and 'E5_E2_plus_OWID_lag1y' in ablation_test_predictions:
    add_dm('E8_OWID_lag2 vs E5_OWID_lag1',ablation_test_predictions['E8_E2_plus_OWID_lag2y_publication_sensitivity'],ablation_test_predictions['E5_E2_plus_OWID_lag1y'])

stat_tests_df=pd.DataFrame(comparisons)
primary=stat_tests_df[stat_tests_df['primary_HAC_specification']].copy()
if not primary.empty:
    reject,padj,_,_=multipletests(primary['p_two_sided'].values,alpha=0.05,method='holm')
    primary['p_Holm']=padj; primary['reject_H0_Holm_0_05']=reject
    stat_tests_df=stat_tests_df.merge(primary[['comparison','HAC_lags','p_Holm','reject_H0_Holm_0_05']],on=['comparison','HAC_lags'],how='left')

# Wilcoxon absolute-error sensitivity for the main baseline comparisons.
wilcoxon_rows=[]
for b in ['Persistence_lag1','SeasonalNaive_lag168','XGBoost_default']:
    if b in benchmark_predictions:
        eA=np.abs(y_test.values-final_test_pred); eB=np.abs(y_test.values-benchmark_predictions[b])
        try:
            w=wilcoxon(eA,eB,alternative='less')
            wilcoxon_rows.append({'comparison':f'{selected_name} vs {b}','test':'Wilcoxon absolute error, A<B','statistic':float(w.statistic),'p_value':float(w.pvalue)})
        except Exception as e: wilcoxon_rows.append({'comparison':f'{selected_name} vs {b}','test':'Wilcoxon','error':repr(e)})
wilcoxon_df=pd.DataFrame(wilcoxon_rows)

display(stat_tests_df)
display(wilcoxon_df)
