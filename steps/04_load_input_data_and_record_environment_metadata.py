# Step 04: Load input data and record environment metadata

print_section('INPUT FILES AND REPRODUCIBILITY METADATA')

if not HOURLY_FILE.exists():
    raise FileNotFoundError(f'Missing hourly input: {HOURLY_FILE}')
if not OWID_FILE.exists():
    raise FileNotFoundError(f'Missing OWID input: {OWID_FILE}')

hourly_raw = pd.read_csv(HOURLY_FILE)
owid_raw = pd.read_csv(OWID_FILE)
hourly = sanitize_columns(hourly_raw)
owid = sanitize_columns(owid_raw)

package_names = ['numpy','pandas','sklearn','xgboost','lightgbm','catboost','shap','holidays','optuna','statsmodels']
package_rows=[]
for name in package_names:
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
    except Exception as e:
        version = f'not_available: {e}'
    package_rows.append({'package':name,'version':version})
package_versions_df = pd.DataFrame(package_rows)

data_fingerprints_df = pd.DataFrame([
    {'file_role':'hourly_market_data','path':str(HOURLY_FILE),'bytes':HOURLY_FILE.stat().st_size,'sha256':sha256_file(HOURLY_FILE)},
    {'file_role':'owid_energy_data','path':str(OWID_FILE),'bytes':OWID_FILE.stat().st_size,'sha256':sha256_file(OWID_FILE)},
])

proc = psutil.Process(os.getpid())
hardware_df = pd.DataFrame([
    {'item':'python_version','value':sys.version.split()[0]},
    {'item':'platform','value':platform.platform()},
    {'item':'cpu_count_logical','value':psutil.cpu_count(logical=True)},
    {'item':'cpu_count_physical','value':psutil.cpu_count(logical=False)},
    {'item':'ram_total_GB','value':round(psutil.virtual_memory().total/1024**3,3)},
    {'item':'initial_process_RSS_MB','value':round(proc.memory_info().rss/1024**2,3)},
    {'item':'run_timestamp_UTC','value':datetime.utcnow().isoformat()},
])

print('Hourly raw shape:', hourly.shape)
print('OWID raw shape:', owid.shape)
display(data_fingerprints_df)
display(package_versions_df)
