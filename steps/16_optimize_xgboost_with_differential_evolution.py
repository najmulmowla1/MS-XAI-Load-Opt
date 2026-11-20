# Step 16: Optimize XGBoost with differential evolution

print_section('OPTIMIZATION — DIFFERENTIAL EVOLUTION')

# Custom DE is used so "population size = 8" means exactly eight parameter vectors,
# rather than scipy's popsize multiplier-by-dimensionality convention.
DE_KEYS=['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','reg_alpha','reg_lambda','gamma']
DE_BOUNDS=np.array([
    [np.log(0.01),np.log(0.15)], [3,10], [np.log(1),np.log(20)], [0.60,1.00], [0.60,1.00],
    [np.log(1e-4),np.log(10)], [np.log(1e-3),np.log(30)], [0,5]
],dtype=float)

def decode_de(x):
    return {
        'learning_rate':float(np.exp(x[0])),
        'max_depth':int(np.clip(round(x[1]),3,10)),
        'min_child_weight':float(np.exp(x[2])),
        'subsample':float(x[3]),'colsample_bytree':float(x[4]),
        'reg_alpha':float(np.exp(x[5])),'reg_lambda':float(np.exp(x[6])),'gamma':float(x[7])
    }

if RUN_DIFFERENTIAL_EVOLUTION:
    rng_de=np.random.default_rng(RANDOM_STATE+17)
    dim=len(DE_KEYS); pop_n=DE_POPULATION_SIZE
    pop=rng_de.uniform(DE_BOUNDS[:,0],DE_BOUNDS[:,1],size=(pop_n,dim))
    scores=np.empty(pop_n)
    de_trial=0
    for i in range(pop_n):
        de_trial+=1; p=decode_de(pop[i]); scores[i],sd=cv_score_xgb_params(p)
        optimizer_trials.append({'optimizer':'DifferentialEvolution_XGBoost','trial':de_trial,'mean_CV_RMSE_MWh':scores[i],'std_CV_RMSE_MWh':sd,'params_json':safe_json(p)})
    F=0.7; CR=0.8
    for it in range(1,DE_ITERATIONS+1):
        for i in range(pop_n):
            choices=[j for j in range(pop_n) if j!=i]
            a,b,c=rng_de.choice(choices,3,replace=False)
            mutant=pop[a]+F*(pop[b]-pop[c])
            mutant=np.clip(mutant,DE_BOUNDS[:,0],DE_BOUNDS[:,1])
            cross=rng_de.random(dim)<CR
            if not cross.any(): cross[rng_de.integers(0,dim)]=True
            trial_vec=np.where(cross,mutant,pop[i])
            de_trial+=1; p=decode_de(trial_vec); score,sd=cv_score_xgb_params(p)
            optimizer_trials.append({'optimizer':'DifferentialEvolution_XGBoost','trial':de_trial,'mean_CV_RMSE_MWh':score,'std_CV_RMSE_MWh':sd,'params_json':safe_json(p)})
            if score<scores[i]: pop[i]=trial_vec; scores[i]=score
        print(f'DE iteration {it}/{DE_ITERATIONS}: best mean CV RMSE={scores.min():.3f}')
    bi=int(np.argmin(scores)); bp=decode_de(pop[bi]); bm,bsd=cv_score_xgb_params(bp)
    optimized_candidates['XGBoost_DifferentialEvolution']={'model_family':'XGBoost','optimizer':'DifferentialEvolution','params':bp,
        'mean_CV_RMSE_MWh':bm,'std_CV_RMSE_MWh':bsd}

optimizer_trials_df=pd.DataFrame(optimizer_trials)
optimization_candidates_df=pd.DataFrame([
    {'candidate_model':k,'model_family':v['model_family'],'optimizer':v['optimizer'],
     'mean_CV_RMSE_MWh':v['mean_CV_RMSE_MWh'],'std_CV_RMSE_MWh':v['std_CV_RMSE_MWh'],'params_json':safe_json(v['params'])}
    for k,v in optimized_candidates.items()
]).sort_values('mean_CV_RMSE_MWh').reset_index(drop=True)

display(optimization_candidates_df)
save_csv(optimizer_trials_df,'optimizer_trials_all')
save_csv(optimization_candidates_df,'optimization_candidates')
