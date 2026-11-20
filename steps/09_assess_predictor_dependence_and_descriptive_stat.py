# Step 09: Assess predictor dependence and descriptive statistics

print_section('CORRELATION / REDUNDANCY / DESCRIPTIVE DIAGNOSTICS')

pretest_mask = model_df['target_year'] <= PRETEST_END_YEAR
corr_input = model_df.loc[pretest_mask, FULL_FEATURES].astype(float)
feature_correlation_df = corr_input.corr(method='pearson')

# Top unique pairwise correlations.
rows=[]
cols=list(feature_correlation_df.columns)
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        v=feature_correlation_df.iat[i,j]
        if np.isfinite(v):
            rows.append({'feature_1':cols[i],'feature_2':cols[j],'correlation':float(v),'abs_correlation':abs(float(v))})
top_correlations_df = pd.DataFrame(rows).sort_values('abs_correlation',ascending=False).reset_index(drop=True)

redundancy_summary=[]
for c in cols:
    vals=feature_correlation_df[c].drop(index=c).abs()
    redundancy_summary.append({
        'feature':c,
        'max_abs_corr_with_other':float(vals.max()),
        'count_abs_corr_ge_0_90':int((vals>=0.90).sum()),
        'count_abs_corr_ge_0_95':int((vals>=0.95).sum()),
    })
redundancy_summary_df = pd.DataFrame(redundancy_summary).sort_values('max_abs_corr_with_other',ascending=False)

# Target descriptive summaries.
yearly_target_summary_df = model_df.groupby('target_year')[TARGET].agg(['count','mean','std','min','max']).reset_index()
hourly_profile_df = model_df.groupby('hour')[TARGET].agg(['mean','std','count']).reset_index()
monthly_hourly_profile_df = model_df.groupby('month')[TARGET].agg(['mean','std','count']).reset_index()
monthly_total_gwh_df = (model_df.set_index('target_timestamp')[TARGET].resample('MS').sum()/1000).rename('monthly_total_GWh').reset_index()

# Correlation heatmap: top features by correlation with target lag-1 and variance, capped for readability.
heat_features = list(dict.fromkeys(
    TARGET_LAG_COLS + ROLLING_COLS[:8] + GEN_LAG_COLS[:12] + PRICE_LAG_COLS + OWID_LAG1_COLS[:8] + TEMPORAL_COLS
))[:45]
plt.figure(figsize=(16,13))
plt.imshow(model_df.loc[pretest_mask, heat_features].corr().values, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label='Pearson correlation')
plt.xticks(range(len(heat_features)), heat_features, rotation=90, fontsize=7)
plt.yticks(range(len(heat_features)), heat_features, fontsize=7)
plt.title('Selected Predictor Correlation Matrix — Pre-test Data (2018–2022)')
plt.tight_layout()
CORR_FIG = FIG_DIR / 'correlation_matrix_selected_features.png'
plt.savefig(CORR_FIG,dpi=300,bbox_inches='tight'); plt.show()

display(top_correlations_df.head(30))
