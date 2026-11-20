# Step 06: Audit annual energy indicators

print_section('OWID TURKEY AUDIT')

if 'country' not in owid.columns or 'year' not in owid.columns:
    raise ValueError("OWID must contain 'country' and 'year'.")

labels = owid.loc[owid['country'].astype(str).str.lower().isin(['turkey','türkiye']), 'country'].unique()
if len(labels)==0:
    raise ValueError('Turkey/Türkiye not found in OWID country column.')
TURKEY_LABEL = 'Turkey' if 'Turkey' in labels else labels[0]
owid_tr = owid[owid['country']==TURKEY_LABEL].copy()
owid_tr['year'] = pd.to_numeric(owid_tr['year'], errors='coerce').astype('Int64')
owid_tr = owid_tr.sort_values('year').reset_index(drop=True)

OWID_CANDIDATES = [
    'population','gdp','primary_energy_consumption','energy_per_capita','energy_per_gdp',
    'electricity_generation','electricity_demand','per_capita_electricity',
    'fossil_fuel_consumption','renewables_consumption','low_carbon_consumption',
    'fossil_share_energy','renewables_share_energy','low_carbon_share_energy',
    'fossil_share_elec','renewables_share_elec','low_carbon_share_elec',
    'carbon_intensity_elec','greenhouse_gas_emissions','coal_electricity','gas_electricity',
    'oil_electricity','hydro_electricity','solar_electricity','wind_electricity','nuclear_electricity'
]
available_candidates = [c for c in OWID_CANDIDATES if c in owid_tr.columns]
for c in available_candidates:
    owid_tr[c] = pd.to_numeric(owid_tr[c], errors='coerce')

# Main + publication-lag sensitivity require 2016-2022.
required_years = list(range(2016, PRETEST_END_YEAR+1))
req = owid_tr[owid_tr['year'].isin(required_years)].copy()

inventory=[]
for c in OWID_CANDIDATES:
    exists = c in owid_tr.columns
    if exists:
        miss_years = req.loc[req[c].isna(),'year'].astype(str).tolist()
        complete = len(miss_years)==0 and set(required_years).issubset(set(req['year'].dropna().astype(int)))
        inventory.append({'variable':c,'exists':True,'required_years':f'{min(required_years)}-{max(required_years)}',
                          'missing_required_years':','.join(miss_years),'retained_complete_case':bool(complete)})
    else:
        inventory.append({'variable':c,'exists':False,'required_years':f'{min(required_years)}-{max(required_years)}',
                          'missing_required_years':'column_absent','retained_complete_case':False})
owid_inventory_df = pd.DataFrame(inventory)

# Retained annual indicators are not interpolated or backfilled.
# Only indicators complete over all years needed for lag-1 and lag-2 sensitivity are retained.
OWID_FEATURES = owid_inventory_df.loc[owid_inventory_df['retained_complete_case'],'variable'].tolist()
if len(OWID_FEATURES)==0:
    raise ValueError('No OWID candidate has complete coverage over the required years.')

owid_required_values_df = req[['year']+OWID_FEATURES].copy()
owid_policy_df = pd.DataFrame([
    {'item':'country_filter','value':TURKEY_LABEL},
    {'item':'candidate_indicator_count','value':len(OWID_CANDIDATES)},
    {'item':'available_candidate_count','value':len(available_candidates)},
    {'item':'retained_complete_indicator_count','value':len(OWID_FEATURES)},
    {'item':'required_years','value':f'{min(required_years)}-{max(required_years)}'},
    {'item':'missing_value_policy','value':'retain only indicators complete over required years; no interpolation, no forward fill, no backward fill'},
    {'item':'main_merge','value':'target year Y receives OWID Y-1'},
    {'item':'publication_lag_sensitivity','value':'target year Y also evaluated with OWID Y-2'},
])

display(owid_policy_df)
display(owid_inventory_df)
print('Retained OWID features:', OWID_FEATURES)

save_csv(owid_inventory_df,'owid_inventory')
save_csv(owid_required_values_df,'owid_required_values')
