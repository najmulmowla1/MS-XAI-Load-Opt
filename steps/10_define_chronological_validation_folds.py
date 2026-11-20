# Step 10: Define chronological validation folds

print_section('TEMPORAL VALIDATION DESIGN')

CV_FOLDS=[]
fold_rows=[]
for val_year in CV_VALIDATION_YEARS:
    train_idx = model_df.index[model_df['target_year'] < val_year].to_numpy()
    val_idx = model_df.index[model_df['target_year'] == val_year].to_numpy()
    if len(train_idx)==0 or len(val_idx)==0:
        continue
    fold_id=f'train_through_{val_year-1}_validate_{val_year}'
    CV_FOLDS.append({'fold':fold_id,'train_idx':train_idx,'val_idx':val_idx,'val_year':val_year})
    fold_rows.append({
        'fold':fold_id,'train_start':str(model_df.loc[train_idx,'target_timestamp'].min()),
        'train_end':str(model_df.loc[train_idx,'target_timestamp'].max()),
        'validation_start':str(model_df.loc[val_idx,'target_timestamp'].min()),
        'validation_end':str(model_df.loc[val_idx,'target_timestamp'].max()),
        'n_train':len(train_idx),'n_validation':len(val_idx)
    })
cv_folds_df=pd.DataFrame(fold_rows)

PRETEST_IDX=model_df.index[model_df['target_year']<=PRETEST_END_YEAR].to_numpy()
TEST_IDX=model_df.index[model_df['target_year']==TEST_YEAR].to_numpy()
if len(TEST_IDX)==0:
    raise ValueError(f'No rows found for independent test year {TEST_YEAR}.')

split_summary_df=pd.DataFrame([
    {'split':'pretest_all','period':f'<= {PRETEST_END_YEAR}','n':len(PRETEST_IDX)},
    {'split':'independent_test','period':str(TEST_YEAR),'n':len(TEST_IDX)},
])

display(cv_folds_df)
display(split_summary_df)
print('🔒 2023 is not used in any CV fold or optimizer objective.')
