# Step 21: Specify deep-learning benchmark settings

print_section('DEEP LEARNING BENCHMARK CONFIGURATION')

dl_config_df=pd.DataFrame([
    {'item':'sequence_input','value':f'univariate historical consumption only, previous {DL_WINDOW} hours'},
    {'item':'forecast_horizon','value':'one hour ahead'},
    {'item':'normalization','value':'z-score using 2018–2021 training mean/std; same scale applied to validation/test'},
    {'item':'early_stopping_validation','value':'2022'},
    {'item':'max_epochs','value':DL_MAX_EPOCHS},
    {'item':'patience','value':DL_PATIENCE},
    {'item':'batch_size','value':DL_BATCH_SIZE},
    {'item':'optimizer','value':'Adam'},
    {'item':'initial_learning_rate','value':0.001},
    {'item':'loss','value':'MSE'},
    {'item':'random_seed','value':RANDOM_STATE},
    {'item':'final_training','value':'rebuild and fit 2018–2022 for the best epoch determined without 2023'},
])
display(dl_config_df)
