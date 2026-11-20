#!/usr/bin/env python3
"""Verify exact source CSV snapshots for MS-XAI-Load-Opt reproduction.

Uses only the Python standard library. Does not fit models or alter data.
"""
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'metadata' / 'dataset_fingerprints.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(path, spec, name):
    errors = []
    actual_hash = sha256_file(path)
    if actual_hash != spec['sha256']:
        errors.append('SHA-256 fingerprint differs from the recorded completed run')
    if path.stat().st_size != spec['bytes']:
        errors.append('File size differs from the recorded completed run')
    with path.open(encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        fields = rdr.fieldnames or []
        if len(fields) != spec['columns']:
            errors.append('Column count mismatch')
        rows = 0
        if name == 'hourly_energy_data.csv':
            timestamps = []
            missing_target = 0
            for row in rdr:
                rows += 1
                if not row.get('consumption_MWh', '').strip():
                    missing_target += 1
                try:
                    timestamps.append(datetime.strptime(row['time'], '%d:%m:%Y:%H:%M'))
                except (ValueError, KeyError, TypeError):
                    errors.append(f'Invalid timestamp at data row {rows}')
                    break
            if not errors:
                timestamps.sort()
                if not timestamps or timestamps[0].strftime('%d:%m:%Y:%H:%M') != spec['first_timestamp'] or timestamps[-1].strftime('%d:%m:%Y:%H:%M') != spec['last_timestamp']:
                    errors.append('Hourly timestamp coverage mismatch')
                if len(timestamps) != len(set(timestamps)):
                    errors.append('Duplicate hourly timestamps')
                if len(timestamps) > 1 and any((b-a) != timedelta(hours=1) for a,b in zip(timestamps,timestamps[1:])):
                    errors.append('Hourly time series is not continuous')
            if missing_target:
                errors.append(f'{missing_target} target observations missing')
        else:
            turkey = []
            for row in rdr:
                rows += 1
                if row.get('country','').lower() in ('turkey','türkiye'):
                    turkey.append(row)
            years = {int(row['year']) for row in turkey}
            if len(turkey) != spec['target_country_rows']:
                errors.append('Türkiye country-row count mismatch')
            if not turkey or [min(years),max(years)] != spec['target_country_years']:
                errors.append('Türkiye country-year coverage mismatch')
            expected_range = set(range(2016,2023))
            candidate_cols = ['population','gdp','primary_energy_consumption','energy_per_capita','energy_per_gdp',
              'electricity_generation','electricity_demand','per_capita_electricity','fossil_fuel_consumption',
              'renewables_consumption','low_carbon_consumption','fossil_share_energy','renewables_share_energy',
              'low_carbon_share_energy','fossil_share_elec','renewables_share_elec','low_carbon_share_elec',
              'carbon_intensity_elec','greenhouse_gas_emissions','coal_electricity','gas_electricity',
              'oil_electricity','hydro_electricity','solar_electricity','wind_electricity','nuclear_electricity']
            complete = [col for col in candidate_cols if col in fields and all(r.get(col,'').strip() not in ('','nan') for r in turkey if int(r['year']) in expected_range)]
            if not expected_range.issubset(years) or len(complete) != 26:
                errors.append(f'OWID pretest indicator coverage mismatch: {len(complete)} complete indicators')
        if rows != spec['rows']:
            errors.append(f'Row count mismatch: {rows} instead of {spec["rows"]}')
    return {'file':name,'rows':rows,'columns':len(fields),'sha256':actual_hash,'status':'PASS' if not errors else 'FAIL','issues':errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--json-report',type=Path,default=None)
    args=parser.parse_args()
    manifest=json.loads(DEFAULT_MANIFEST.read_text(encoding='utf-8'))
    results=[]
    for name,spec in manifest['datasets'].items():
        path=args.data_dir.expanduser().resolve()/name
        results.append(audit_dataset(path,spec,name) if path.is_file() else {'file':name,'status':'FAIL','issues':['File does not exist']})
    report={'result':'PASS' if all(x['status']=='PASS' for x in results) else 'FAIL','inputs':results}
    if args.json_report:
        args.json_report.parent.mkdir(parents=True,exist_ok=True)
        args.json_report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['result']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
