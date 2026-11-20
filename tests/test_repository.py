"""Source integrity tests; no input datasets or expensive model fitting required."""
import ast
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class TestRepository(unittest.TestCase):
    def test_manifest(self):
        data = json.loads((ROOT/'metadata'/'cell_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data), 33)
        self.assertEqual(len([x for x in data if x['type']=='code']), 32)
        self.assertTrue(all((ROOT/x['script']).exists() for x in data if x['type']=='code'))

    def test_scripts_compile(self):
        scripts = sorted((ROOT/'steps').glob('*.py'))
        self.assertEqual(len(scripts), 31)
        for script in scripts:
            with self.subTest(script=script.name):
                ast.parse(script.read_text(encoding='utf-8'))

    def test_scripts_only_release(self):
        self.assertEqual(list(ROOT.rglob('*.ipynb')), [])
        self.assertFalse((ROOT/'notebooks').exists())
        self.assertEqual(list(ROOT.rglob('*.csv')), [])
        manifest = json.loads((ROOT/'metadata'/'cell_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len([item for item in manifest if item['type']=='markdown']), 1)
        self.assertTrue((ROOT/'steps'/'01_install_analysis_dependencies.sh').exists())

    def test_config_and_summary(self):
        summary = json.loads((ROOT/'metadata'/'reported_run_summary.json').read_text(encoding='utf-8'))
        config = json.loads((ROOT/'metadata'/'reported_run_configuration.json').read_text(encoding='utf-8'))
        self.assertEqual(config['cv_validation_years'],[2020,2021,2022])
        self.assertEqual(summary['main_feature_count'],133)
        self.assertAlmostEqual(summary['test_RMSE_MWh'],441.85269050860296)

    def test_data_fingerprints(self):
        fingerprints = json.loads((ROOT/'metadata'/'dataset_fingerprints.json').read_text(encoding='utf-8'))
        datasets = fingerprints['datasets']
        self.assertEqual(set(datasets), {'hourly_energy_data.csv','owid_energy_data.csv'})
        for info in datasets.values():
            self.assertRegex(info['sha256'], r'^[0-9a-f]{64}$')
            self.assertGreater(info['rows'],0)

    def test_no_editorial_language(self):
        code = '\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'steps').glob('*.py'))
        forbidden = re.compile(r'major[\s_-]?revision|reviewer[\s_-]?to[\s_-]?evidence|requested by reviewer',re.I)
        self.assertIsNone(forbidden.search(code))

if __name__=='__main__':
    unittest.main()
