# Step 11: Define model factories and hyperparameter searches

print_section('MODEL DEFINITIONS AND SEARCH SPACES')

def make_ridge():
    return Pipeline([('scaler',StandardScaler()),('model',Ridge(alpha=1.0))])

def make_rf():
    return RandomForestRegressor(n_estimators=350,max_depth=None,min_samples_leaf=1,n_jobs=-1,random_state=RANDOM_STATE)

def make_xgb(params=None):
    base=dict(n_estimators=N_ESTIMATORS_DEFAULT,learning_rate=0.05,max_depth=6,min_child_weight=1,
              subsample=0.9,colsample_bytree=0.9,reg_alpha=0.0,reg_lambda=1.0,gamma=0.0,
              objective='reg:squarederror',tree_method='hist',n_jobs=-1,random_state=RANDOM_STATE,verbosity=0)
    if params: base.update(params)
    return XGBRegressor(**base)

def make_lgbm(params=None):
    base=dict(n_estimators=N_ESTIMATORS_DEFAULT,learning_rate=0.05,num_leaves=31,max_depth=-1,
              min_child_samples=20,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.0,reg_lambda=0.0,
              random_state=RANDOM_STATE,n_jobs=-1,verbosity=-1)
    if params: base.update(params)
    return LGBMRegressor(**base)

def make_catboost():
    return CatBoostRegressor(iterations=500,learning_rate=0.05,depth=8,loss_function='RMSE',
                             random_seed=RANDOM_STATE,verbose=False,allow_writing_files=False)

def make_mlp():
    return Pipeline([
        ('scaler',StandardScaler()),
        ('model',MLPRegressor(hidden_layer_sizes=(128,64),activation='relu',solver='adam',alpha=1e-4,
                              learning_rate_init=1e-3,max_iter=250,early_stopping=True,validation_fraction=0.10,
                              n_iter_no_change=15,random_state=RANDOM_STATE))
    ])

DEFAULT_MODEL_FACTORIES = {
    'Ridge':make_ridge,
    'RandomForest':make_rf,
    'XGBoost_default':lambda:make_xgb(),
    'LightGBM_default':lambda:make_lgbm(),
    'CatBoost_default':make_catboost,
    'MLP_tabular':make_mlp,
}

SEARCH_SPACE_ROWS = [
    {'model':'XGBoost','parameter':'learning_rate','distribution/range':'log-uniform 0.01–0.15'},
    {'model':'XGBoost','parameter':'max_depth','distribution/range':'integer 3–10'},
    {'model':'XGBoost','parameter':'min_child_weight','distribution/range':'log-uniform 1–20'},
    {'model':'XGBoost','parameter':'subsample','distribution/range':'uniform 0.60–1.00'},
    {'model':'XGBoost','parameter':'colsample_bytree','distribution/range':'uniform 0.60–1.00'},
    {'model':'XGBoost','parameter':'reg_alpha','distribution/range':'log-uniform 1e-4–10'},
    {'model':'XGBoost','parameter':'reg_lambda','distribution/range':'log-uniform 1e-3–30'},
    {'model':'XGBoost','parameter':'gamma','distribution/range':'uniform 0–5'},
    {'model':'XGBoost','parameter':'n_estimators','distribution/range':f'fixed {N_ESTIMATORS_OPT}'},
    {'model':'LightGBM','parameter':'learning_rate','distribution/range':'log-uniform 0.01–0.15'},
    {'model':'LightGBM','parameter':'num_leaves','distribution/range':'integer 15–127'},
    {'model':'LightGBM','parameter':'max_depth','distribution/range':'integer 4–12'},
    {'model':'LightGBM','parameter':'min_child_samples','distribution/range':'integer 10–100'},
    {'model':'LightGBM','parameter':'subsample','distribution/range':'uniform 0.60–1.00'},
    {'model':'LightGBM','parameter':'colsample_bytree','distribution/range':'uniform 0.60–1.00'},
    {'model':'LightGBM','parameter':'reg_alpha','distribution/range':'log-uniform 1e-4–10'},
    {'model':'LightGBM','parameter':'reg_lambda','distribution/range':'log-uniform 1e-3–30'},
    {'model':'LightGBM','parameter':'n_estimators','distribution/range':f'fixed {N_ESTIMATORS_OPT}'},
]
search_spaces_df=pd.DataFrame(SEARCH_SPACE_ROWS)

# Cache fold data once for the main full feature matrix.
CV_DATA={}
for fold in CV_FOLDS:
    tr,va=fold['train_idx'],fold['val_idx']
    CV_DATA[fold['fold']]={
        'X_train':model_df.loc[tr,FULL_FEATURES].astype(np.float32),
        'y_train':model_df.loc[tr,TARGET].astype(float),
        'X_val':model_df.loc[va,FULL_FEATURES].astype(np.float32),
        'y_val':model_df.loc[va,TARGET].astype(float),
        'val_idx':va,
    }

def cv_evaluate_factory(factory, features=FULL_FEATURES, model_name='model', return_oof=False):
    rows=[]; oof=[]
    for fold in CV_FOLDS:
        tr,va=fold['train_idx'],fold['val_idx']
        Xtr=model_df.loc[tr,features].astype(np.float32); ytr=model_df.loc[tr,TARGET]
        Xva=model_df.loc[va,features].astype(np.float32); yva=model_df.loc[va,TARGET]
        model=factory()
        t0=time.perf_counter(); model.fit(Xtr,ytr); fit_s=time.perf_counter()-t0
        p_tr=model.predict(Xtr); p_va=model.predict(Xva)
        tr_rmse=rmse(ytr,p_tr); va_metrics=evaluate_forecast(yva,p_va,mase_scale_from_training(ytr.values))
        row={'model':model_name,'fold':fold['fold'],'val_year':fold['val_year'],'train_RMSE_MWh':tr_rmse,
             'generalization_gap_RMSE_MWh':va_metrics['RMSE_MWh']-tr_rmse,'fit_seconds':fit_s,**va_metrics}
        rows.append(row)
        if return_oof:
            oof.append(pd.DataFrame({'row_index':va,'target_timestamp':model_df.loc[va,'target_timestamp'].values,
                                     'y_true':yva.values,'y_pred':p_va,'fold':fold['fold']}))
        del model,Xtr,Xva; gc.collect()
    result=pd.DataFrame(rows)
    if return_oof:
        return result,pd.concat(oof,ignore_index=True)
    return result

def cv_score_xgb_params(params):
    fold_rmse=[]
    for fold in CV_FOLDS:
        d=CV_DATA[fold['fold']]
        m=make_xgb({**params,'n_estimators':N_ESTIMATORS_OPT})
        m.fit(d['X_train'],d['y_train'])
        fold_rmse.append(rmse(d['y_val'],m.predict(d['X_val'])))
        del m; gc.collect()
    return float(np.mean(fold_rmse)), float(np.std(fold_rmse,ddof=1) if len(fold_rmse)>1 else 0.0)

def cv_score_lgbm_params(params):
    fold_rmse=[]
    for fold in CV_FOLDS:
        d=CV_DATA[fold['fold']]
        m=make_lgbm({**params,'n_estimators':N_ESTIMATORS_OPT})
        m.fit(d['X_train'],d['y_train'])
        fold_rmse.append(rmse(d['y_val'],m.predict(d['X_val'])))
        del m; gc.collect()
    return float(np.mean(fold_rmse)), float(np.std(fold_rmse,ddof=1) if len(fold_rmse)>1 else 0.0)

display(search_spaces_df)
