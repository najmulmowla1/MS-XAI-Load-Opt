# Step 32: Package generated results

print_section('PACKAGE OUTPUTS')

import shutil
zip_base=str(OUTPUT_DIR.parent / 'MS_XAI_Load_Opt_Evidence')
zip_path=shutil.make_archive(zip_base,'zip',root_dir=str(OUTPUT_DIR))
print('MASTER Excel:',MASTER_EXCEL_PATH)
print('Evidence ZIP:',zip_path)
print('Selected model:',selected_name)
print('2023 RMSE (MWh):',final_test_metrics['RMSE_MWh'])
print('2023 MAPE (%):',final_test_metrics['MAPE_pct'])

