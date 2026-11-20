# Step 27: Explain the selected model using SHAP

print_section('SHAP — ACTUAL SELECTED MODEL')

shap_feature_importance_df=pd.DataFrame(); shap_group_importance_df=pd.DataFrame(); shap_errors_df=pd.DataFrame(); shap_values_sample_df=pd.DataFrame()
try:
    import shap
    X_shap=X_test.sample(n=min(SHAP_SAMPLE_SIZE,len(X_test)),random_state=RANDOM_STATE)
    explainer=shap.TreeExplainer(final_model)
    sv=explainer.shap_values(X_shap)
    if isinstance(sv,list): sv=sv[0]
    sv=np.asarray(sv)
    mean_abs=np.abs(sv).mean(axis=0)
    shap_values_sample_df=pd.DataFrame(sv,columns=X_shap.columns)
    shap_values_sample_df.insert(0,'sample_row_index',X_shap.index.values)
    shap_feature_importance_df=pd.DataFrame({'feature':X_shap.columns,'mean_abs_SHAP':mean_abs}).sort_values('mean_abs_SHAP',ascending=False).reset_index(drop=True)
    group_map={}
    for g,cols_ in FEATURE_GROUPS.items():
        for c in cols_: group_map[c]=g
    shap_feature_importance_df['feature_group']=shap_feature_importance_df['feature'].map(group_map).fillna('other')
    shap_group_importance_df=(shap_feature_importance_df.groupby('feature_group',as_index=False)['mean_abs_SHAP'].sum()
                              .sort_values('mean_abs_SHAP',ascending=False).reset_index(drop=True))
    shap_group_importance_df['share_pct']=100*shap_group_importance_df['mean_abs_SHAP']/shap_group_importance_df['mean_abs_SHAP'].sum()

    plt.figure(); shap.summary_plot(sv,X_shap,show=False,max_display=25); plt.tight_layout()
    SHAP_BEES=FIG_DIR/'shap_selected_model_beeswarm.png'; plt.savefig(SHAP_BEES,dpi=300,bbox_inches='tight'); plt.show()
    plt.figure(); shap.summary_plot(sv,X_shap,plot_type='bar',show=False,max_display=25); plt.tight_layout()
    SHAP_BAR=FIG_DIR/'shap_selected_model_bar.png'; plt.savefig(SHAP_BAR,dpi=300,bbox_inches='tight'); plt.show()
    print('SHAP model:',selected_name)
    display(shap_group_importance_df)
except Exception as e:
    shap_errors_df=pd.DataFrame([{'error':repr(e)}]); print('SHAP failed:',repr(e))
