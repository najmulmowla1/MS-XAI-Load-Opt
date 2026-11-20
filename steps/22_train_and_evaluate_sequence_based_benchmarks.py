# Step 22: Train and evaluate sequence-based benchmarks

print_section('DEEP LEARNING BENCHMARKS')

dl_results=[]; dl_errors=[]
if RUN_DEEP_LEARNING:
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        try: tf.config.experimental.enable_op_determinism()
        except Exception: pass
        from tensorflow.keras import layers, models, callbacks, optimizers

        raw_seq=df[['target_timestamp',TARGET]].dropna().sort_values('target_timestamp').reset_index(drop=True)
        vals=raw_seq[TARGET].astype(np.float32).values
        times=raw_seq['target_timestamp'].values
        # Windows ending immediately before each target observation.
        X_seq=np.lib.stride_tricks.sliding_window_view(vals,DL_WINDOW)[:-1].copy()
        y_seq=vals[DL_WINDOW:].copy()
        t_seq=pd.to_datetime(times[DL_WINDOW:])
        years=t_seq.year
        train_mask=years<=2021; val_mask=years==2022; test_mask=years==2023; pre_mask=years<=2022
        mu=float(y_seq[train_mask].mean()); sd=float(y_seq[train_mask].std())
        Xs=(X_seq-mu)/sd; ys=(y_seq-mu)/sd
        Xs=Xs[...,None].astype(np.float32); ys=ys.astype(np.float32)

        def transformer_block(x,d_model=32,num_heads=4,ff_dim=64,dropout=0.1):
            attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model//num_heads,dropout=dropout)(x,x)
            x=layers.LayerNormalization(epsilon=1e-6)(x+attn)
            ff=layers.Dense(ff_dim,activation='relu')(x); ff=layers.Dropout(dropout)(ff); ff=layers.Dense(d_model)(ff)
            return layers.LayerNormalization(epsilon=1e-6)(x+ff)

        def build_dl(name):
            inp=layers.Input(shape=(DL_WINDOW,1))
            if name=='LSTM':
                x=layers.LSTM(64)(inp); x=layers.Dropout(0.2)(x); x=layers.Dense(32,activation='relu')(x)
            elif name=='GRU':
                x=layers.GRU(64)(inp); x=layers.Dropout(0.2)(x); x=layers.Dense(32,activation='relu')(x)
            elif name=='CNN_LSTM':
                x=layers.Conv1D(32,3,padding='causal',activation='relu')(inp); x=layers.MaxPooling1D(2)(x); x=layers.LSTM(64)(x); x=layers.Dense(32,activation='relu')(x)
            elif name=='Transformer':
                d_model=32; x=layers.Dense(d_model)(inp)
                positions=tf.range(start=0,limit=DL_WINDOW,delta=1)
                pos=layers.Embedding(input_dim=DL_WINDOW,output_dim=d_model)(positions)
                x=x+pos; x=transformer_block(x,d_model=d_model); x=transformer_block(x,d_model=d_model)
                x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(32,activation='relu')(x)
            elif name=='CNN_Transformer':
                d_model=32; x=layers.Conv1D(d_model,5,padding='causal',activation='relu')(inp)
                positions=tf.range(start=0,limit=DL_WINDOW,delta=1)
                pos=layers.Embedding(input_dim=DL_WINDOW,output_dim=d_model)(positions)
                x=x+pos; x=transformer_block(x,d_model=d_model); x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(32,activation='relu')(x)
            else: raise ValueError(name)
            out=layers.Dense(1)(x)
            m=models.Model(inp,out)
            m.compile(optimizer=optimizers.Adam(learning_rate=1e-3),loss='mse')
            return m

        for name in DL_MODELS:
            print('DL benchmark:',name)
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(RANDOM_STATE)
            m=build_dl(name)
            es=callbacks.EarlyStopping(monitor='val_loss',patience=DL_PATIENCE,restore_best_weights=True,min_delta=1e-6)
            rl=callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-5)
            t0=time.perf_counter()
            hist=m.fit(Xs[train_mask],ys[train_mask],validation_data=(Xs[val_mask],ys[val_mask]),
                       epochs=DL_MAX_EPOCHS,batch_size=DL_BATCH_SIZE,verbose=0,callbacks=[es,rl])
            tuning_seconds=time.perf_counter()-t0
            best_epoch=int(np.argmin(hist.history['val_loss'])+1)
            val_pred=(m.predict(Xs[val_mask],batch_size=1024,verbose=0).reshape(-1)*sd+mu)
            val_metrics=evaluate_forecast(y_seq[val_mask],val_pred,mase_scale_from_training(y_seq[train_mask]))

            # Rebuild and train through 2022 for exactly the selected epoch count.
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(RANDOM_STATE)
            m2=build_dl(name); t0=time.perf_counter()
            m2.fit(Xs[pre_mask],ys[pre_mask],epochs=best_epoch,batch_size=DL_BATCH_SIZE,verbose=0)
            final_seconds=time.perf_counter()-t0
            test_pred=(m2.predict(Xs[test_mask],batch_size=1024,verbose=0).reshape(-1)*sd+mu)
            # align sequence test times to the model test year; both contain all 2023 hours
            test_metrics=evaluate_forecast(y_seq[test_mask],test_pred,mase_scale_from_training(y_seq[pre_mask]))
            dl_results.append({'model':name,'family':'deep_learning_sequence','best_epoch':best_epoch,
                               'train_seconds':tuning_seconds+final_seconds,
                               **{f'val_{k}':v for k,v in val_metrics.items()},**test_metrics})
            benchmark_predictions[name]=test_pred
            benchmark_rows.append({'model':name,'family':'deep_learning_sequence','validation_protocol':'2018–2021 train / 2022 early-stopping validation; refit through 2022','train_seconds':tuning_seconds+final_seconds,**test_metrics})
            # Save architecture JSON; final weights optional to reduce storage.
            with open(MODEL_DIR/f'{name}_architecture.json','w') as f: f.write(m2.to_json())
            del m,m2; gc.collect()

    except Exception as e:
        dl_errors.append({'component':'deep_learning_suite','error':repr(e)})
        print('Deep-learning suite failed:',repr(e))

dl_results_df=pd.DataFrame(dl_results)
dl_errors_df=pd.DataFrame(dl_errors)
if not dl_results_df.empty: display(dl_results_df)
