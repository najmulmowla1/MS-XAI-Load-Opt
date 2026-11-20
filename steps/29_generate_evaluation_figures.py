# Step 29: Generate evaluation figures

print_section('PUBLICATION FIGURES')

figure_index=[]
def register(path,description): figure_index.append({'file':str(path),'description':description})
register(CORR_FIG,'Selected predictor correlation matrix on pre-test data')
if 'SHAP_BEES' in globals(): register(SHAP_BEES,'SHAP beeswarm for actual selected optimized model')
if 'SHAP_BAR' in globals(): register(SHAP_BAR,'SHAP global bar plot for actual selected optimized model')

# Data availability timing diagram.
plt.figure(figsize=(13,5))
plt.axvline(0,color='black',linewidth=1); plt.axvline(1,color='black',linewidth=1,linestyle='--')
plt.text(0,4.6,'Forecast origin t',ha='center',fontweight='bold'); plt.text(1,4.6,'Target hour t+1',ha='center',fontweight='bold')
items=[('Historical demand',-1.5,0,4),('Lag-1 demand y(t)',0,0,3),('Target calendar',0,1,2),('Lag-1 generation/price*',0,0,1),('Target y(t+1)',1,1,0)]
for label,x0,x1,y in items:
    plt.plot([x0,x1],[y,y],linewidth=5); plt.text((x0+x1)/2,y+0.15,label,ha='center',fontsize=9)
plt.text(0.5,-0.7,'* Main analysis assumes lag-1 operational data are available at the forecast origin; conservative sensitivity excludes them.',ha='center',fontsize=8)
plt.xlim(-1.7,1.3); plt.ylim(-1,5.1); plt.yticks([]); plt.xticks([-1,0,1],['t−1','t','t+1']); plt.xlabel('Relative hour'); plt.title('One-Hour-Ahead Information Availability and Forecast Timing'); plt.tight_layout()
TIMING_FIG=FIG_DIR/'data_availability_timing_diagram.png'; plt.savefig(TIMING_FIG,dpi=300,bbox_inches='tight'); plt.show(); register(TIMING_FIG,'One-hour-ahead information availability timing diagram')

# CV folds diagram.
plt.figure(figsize=(12,4))
for i,fold in enumerate(CV_FOLDS):
    vy=fold['val_year']; plt.plot([2018,vy-0.1],[i,i],linewidth=8,label='Training' if i==0 else None); plt.plot([vy,vy+0.8],[i,i],linewidth=8,label='Validation' if i==0 else None)
plt.axvspan(2023,2024,alpha=0.15,label='Independent test (unused for selection)')
plt.yticks(range(len(CV_FOLDS)),[f['fold'] for f in CV_FOLDS]); plt.xticks(range(2018,2025)); plt.xlabel('Year'); plt.title('Expanding-Window Temporal Validation Design'); plt.legend(); plt.tight_layout()
CV_FIG=FIG_DIR/'rolling_origin_cv_design.png'; plt.savefig(CV_FIG,dpi=300,bbox_inches='tight'); plt.show(); register(CV_FIG,'Expanding-window temporal validation design')

# Consumption profiles: mean hourly by clock hour; mean hourly by month; monthly total GWh separately.
plt.figure(figsize=(9,5)); plt.plot(hourly_profile_df['hour'],hourly_profile_df['mean']); plt.fill_between(hourly_profile_df['hour'],hourly_profile_df['mean']-hourly_profile_df['std'],hourly_profile_df['mean']+hourly_profile_df['std'],alpha=.2)
plt.xlabel('Hour of day'); plt.ylabel('Mean hourly electricity consumption (MWh)'); plt.title('Mean Hourly Electricity-Consumption Profile'); plt.tight_layout()
P1=FIG_DIR/'mean_hourly_consumption_profile.png'; plt.savefig(P1,dpi=300,bbox_inches='tight'); plt.show(); register(P1,'Mean hourly consumption profile with MWh label')

plt.figure(figsize=(9,5)); plt.plot(monthly_hourly_profile_df['month'],monthly_hourly_profile_df['mean'],marker='o'); plt.xlabel('Month'); plt.ylabel('Mean hourly electricity consumption within month (MWh)'); plt.title('Mean Hourly Consumption by Calendar Month'); plt.xticks(range(1,13)); plt.tight_layout()
P2=FIG_DIR/'mean_hourly_consumption_by_month.png'; plt.savefig(P2,dpi=300,bbox_inches='tight'); plt.show(); register(P2,'Mean hourly consumption within month; avoids confusion with monthly energy total')

plt.figure(figsize=(12,5)); plt.plot(monthly_total_gwh_df['target_timestamp'],monthly_total_gwh_df['monthly_total_GWh']); plt.xlabel('Month'); plt.ylabel('Monthly electricity consumption (GWh)'); plt.title('Monthly Total Electricity Consumption'); plt.tight_layout()
P3=FIG_DIR/'monthly_total_consumption_gwh.png'; plt.savefig(P3,dpi=300,bbox_inches='tight'); plt.show(); register(P3,'Monthly total consumption in GWh')

# Residual and forecast-error diagnostics.
plt.figure(figsize=(10,5)); plt.plot(error_by_hour_df['hour'],error_by_hour_df['MAE_MWh'],marker='o'); plt.xlabel('Hour of day'); plt.ylabel('MAE (MWh)'); plt.title('Forecast Error by Hour of Day — 2023'); plt.tight_layout()
P4=FIG_DIR/'error_by_hour.png'; plt.savefig(P4,dpi=300,bbox_inches='tight'); plt.show(); register(P4,'2023 MAE by hour of day')

plt.figure(figsize=(10,5)); plt.plot(error_by_month_df['month'],error_by_month_df['MAE_MWh'],marker='o'); plt.xlabel('Month'); plt.ylabel('MAE (MWh)'); plt.title('Forecast Error by Month — 2023'); plt.xticks(range(1,13)); plt.tight_layout()
P5=FIG_DIR/'error_by_month.png'; plt.savefig(P5,dpi=300,bbox_inches='tight'); plt.show(); register(P5,'2023 MAE by month')

plt.figure(figsize=(10,5)); plt.scatter(test_results[TARGET],test_results['abs_error_MWh'],s=8,alpha=.25); plt.xlabel('Observed electricity consumption (MWh)'); plt.ylabel('Absolute error (MWh)'); plt.title('Absolute Forecast Error vs Demand Level — 2023'); plt.tight_layout()
P6=FIG_DIR/'error_vs_demand_level.png'; plt.savefig(P6,dpi=300,bbox_inches='tight'); plt.show(); register(P6,'Absolute error versus demand level')

# Conformal calibration curve.
calplot=conformal_summary_df[conformal_summary_df['method']=='simple_split_conformal_OOF']
plt.figure(figsize=(6,6)); plt.plot(calplot['nominal_coverage_pct'],calplot['empirical_coverage_pct'],marker='o',label='Observed'); plt.plot([75,100],[75,100],'--',label='Ideal')
plt.xlabel('Nominal coverage (%)'); plt.ylabel('Empirical coverage (%)'); plt.title('Conformal Calibration — 2023'); plt.legend(); plt.tight_layout()
P7=FIG_DIR/'conformal_calibration_curve.png'; plt.savefig(P7,dpi=300,bbox_inches='tight'); plt.show(); register(P7,'Nominal versus empirical conformal coverage')

# Example interval week: highest-demand week.
weekly=test_results.set_index('target_timestamp')[TARGET].resample('W').max(); end=weekly.idxmax(); start=end-pd.Timedelta(days=7)+pd.Timedelta(hours=1)
g=test_results[(test_results['target_timestamp']>=start)&(test_results['target_timestamp']<=end)]
plt.figure(figsize=(15,5)); plt.plot(g['target_timestamp'],g[TARGET],label='Actual'); plt.plot(g['target_timestamp'],g['prediction'],label='Predicted'); plt.fill_between(g['target_timestamp'],g['lower_90'],g['upper_90'],alpha=.2,label='90% interval')
plt.xlabel('Target hour'); plt.ylabel('Electricity consumption (MWh)'); plt.title('Conformal Interval During a High-Demand Week — 2023'); plt.legend(); plt.tight_layout()
P8=FIG_DIR/'conformal_high_demand_week.png'; plt.savefig(P8,dpi=300,bbox_inches='tight'); plt.show(); register(P8,'Example 90% interval during highest-demand week')


# Pipeline summary diagram.
fig,ax=plt.subplots(figsize=(15,6)); ax.axis('off')
boxes=[
    (0.06,0.72,'Inputs\nHourly market data + annual OWID'),
    (0.28,0.72,'Audit & alignment\nTarget hour t+1 / origin t'),
    (0.50,0.72,'Leakage-safe features\nlagged + rolling + calendar'),
    (0.72,0.72,'Temporal validation\nexpanding-window CV'),
    (0.88,0.42,'Validation-selected\noptimized model'),
    (0.66,0.16,'Post-selection evidence\nSHAP + uncertainty + tests'),
    (0.38,0.16,'Operational diagnostics\nerrors + latency + size'),
    (0.10,0.16,'Independent 2023 test\nused once after selection'),
]
for x,y,text in boxes:
    ax.text(x,y,text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=0.5',fc='white',ec='black'))
arrows=[((0.13,0.72),(0.21,0.72)),((0.35,0.72),(0.43,0.72)),((0.57,0.72),(0.65,0.72)),((0.79,0.70),(0.85,0.48)),
        ((0.84,0.37),(0.72,0.22)),((0.58,0.16),(0.47,0.16)),((0.29,0.16),(0.19,0.16))]
for a,b in arrows:
    ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.5))
ax.set_title('MS-XAI-Load-Opt Forecasting and Evaluation Pipeline',fontsize=14,fontweight='bold')
plt.tight_layout()
PIPE_FIG=FIG_DIR/'pipeline_summary.png'; plt.savefig(PIPE_FIG,dpi=300,bbox_inches='tight'); plt.show(); register(PIPE_FIG,'Forecasting and evaluation pipeline summary')

# Feature-group counts.
fg=feature_group_counts_df[feature_group_counts_df['main_full_features']>0]
plt.figure(figsize=(10,5)); plt.bar(fg['group'],fg['main_full_features']); plt.xticks(rotation=35,ha='right'); plt.ylabel('Number of predictors'); plt.title('Predictor Count by Feature Group — Main Model'); plt.tight_layout()
FG_FIG=FIG_DIR/'feature_group_counts.png'; plt.savefig(FG_FIG,dpi=300,bbox_inches='tight'); plt.show(); register(FG_FIG,'Main-model predictor count by feature group')

# Coverage-width trade-off.
plt.figure(figsize=(7,5)); plt.scatter(conformal_summary_df['mean_width_MWh'],conformal_summary_df['empirical_coverage_pct'])
for _,r in conformal_summary_df.iterrows():
    plt.annotate(f"{r['method']} ({r['nominal_coverage_pct']:.0f}%)",(r['mean_width_MWh'],r['empirical_coverage_pct']),fontsize=7,xytext=(3,3),textcoords='offset points')
plt.xlabel('Mean interval width (MWh)'); plt.ylabel('Empirical coverage (%)'); plt.title('Conformal Coverage–Width Trade-off — 2023'); plt.tight_layout()
CW_FIG=FIG_DIR/'conformal_coverage_width_tradeoff.png'; plt.savefig(CW_FIG,dpi=300,bbox_inches='tight'); plt.show(); register(CW_FIG,'Conformal empirical coverage versus interval width')

for rank in hard_weeks_df.get('rank',[]):
    hp=FIG_DIR/f'hard_week_{int(rank)}.png'
    if hp.exists(): register(hp,f'Hard-week diagnostic {int(rank)}')

figure_index_df=pd.DataFrame(figure_index)
