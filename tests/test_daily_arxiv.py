"""Offline regressions: python -m unittest discover -s tests -v."""
import json
import logging
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import arxiv
import requests
import yaml

import daily_arxiv as daily

ROOT = Path(__file__).resolve().parents[1]
FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
 xmlns:arxiv="http://arxiv.org/schemas/atom">
 <id>test</id><title>test</title><updated>2026-10-29T00:00:00Z</updated>
 <opensearch:totalResults>1</opensearch:totalResults>
 <entry><id>http://arxiv.org/abs/2610.12345v2</id>
 <title>Language Models for Robots</title><summary>A robot planner.</summary>
 <published>2026-10-28T00:00:00Z</published><updated>2026-10-29T00:00:00Z</updated>
 <author><name>First Author</name></author><author><name>Last Author</name></author>
 <category term="cs.RO"/><arxiv:primary_category term="cs.RO"/>
 <arxiv:comment>Code: https://github.com/example/robot</arxiv:comment>
 <link href="http://arxiv.org/abs/2610.12345v2" rel="alternate" type="text/html"/>
 </entry></feed>'''


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def config(self):
        config = daily.load_config(str(ROOT / 'config.yaml'))
        for kind in ('readme', 'gitpage', 'wechat'):
            source = ROOT / config[f'json_{kind}_path']
            target = self.path / source.name
            shutil.copyfile(source, target)
            config[f'json_{kind}_path'] = str(target)
            config[f'md_{kind}_path'] = str(self.path / f'{kind}.md')
        config['update_paper_links'] = False
        return config

    def test_current_arxiv_client_parses_atom_and_formats_both_outputs(self):
        response = Mock(status_code=200, content=FEED)
        client = arxiv.Client(delay_seconds=0)
        with patch.object(client._session, 'get', return_value=response):
            table, web = daily.get_daily_papers('LLM', max_results=1, client=client)
        self.assertEqual(list(table['LLM']), ['2610.12345'])
        self.assertIn('Last Author Team', table['LLM']['2610.12345'])
        self.assertIn('https://github.com/example/robot', web['LLM']['2610.12345'])
        self.assertNotIn('2610.12345v2', table['LLM']['2610.12345'])

    def test_arxiv_http_failure_retries_then_propagates(self):
        client = arxiv.Client(delay_seconds=0, num_retries=2)
        with patch.object(client._session, 'get', return_value=Mock(status_code=503)) as get:
            with self.assertRaises(arxiv.HTTPError):
                daily.get_daily_papers('LLM', client=client)
        self.assertEqual(get.call_count, 3)

    def test_empty_feed_returns_empty_topics(self):
        client = Mock()
        client.results.return_value = iter(())
        self.assertEqual(daily.get_daily_papers('LLM', client=client), ({'LLM': {}}, {'LLM': {}}))

    def test_legacy_filters_have_boolean_separators(self):
        config = daily.load_config(str(ROOT / 'config.yaml'))
        self.assertEqual(config['kv']['Humanoid'], '"Humanoid Robot" OR Humanoid')
        self.assertIn('"Robot Learning" OR "Imitation Learning"', config['kv']['Manipulation'])

    def test_new_queries_preserve_boolean_structure(self):
        config = daily.load_config(str(ROOT / 'config.yaml'))
        for topic in ('LLM-for-Embodied-AI', 'LLM-Technical-Reports'):
            self.assertEqual(config['kv'][topic], config['keywords'][topic]['query'])
            self.assertIn(' AND ', config['kv'][topic])
        self.assertIn('ti:"technical report"', config['kv']['LLM-Technical-Reports'])
        self.assertEqual(len(config['kv']), 11)

    def test_weekly_preserves_wechat_list_entries(self):
        config = self.config()
        path = Path(config['json_wechat_path'])
        original = json.loads(path.read_text())
        daily.update_paper_links(str(path))
        self.assertEqual(json.loads(path.read_text()), original)

    def test_weekly_normalizes_table_versions(self):
        path = self.path / 'papers.json'
        path.write_text(json.dumps({'topic': {'2610.12345': '|date|title|author|[2610.12345v2](http://arxiv.org/abs/2610.12345v2)|null|\n'}}))
        daily.update_paper_links(str(path))
        self.assertNotIn('v2', path.read_text())
        self.assertIn('|null|', path.read_text())

    def test_weekly_preserves_pipes_in_paper_title(self):
        path = self.path / 'papers.json'
        row = '|date|[De|Re]constructing VLMs|author|[2610.12345](http://arxiv.org/abs/2610.12345)|null|\n'
        data = {'VLM': {'2610.12345': row}}
        path.write_text(json.dumps(data))
        daily.update_paper_links(str(path))
        self.assertEqual(json.loads(path.read_text()), data)

    def test_weekly_cli_on_real_history_has_no_git_or_network_side_effects(self):
        config = self.config()
        original = {k: json.loads(Path(config[f'json_{k}_path']).read_text()) for k in ('readme', 'gitpage', 'wechat')}
        config_file = self.path / 'config.yaml'
        config_file.write_text(yaml.safe_dump(config))
        with patch.object(sys, 'argv', ['daily_arxiv.py', '--config_path', str(config_file), '--update_paper_links']), \
             patch('subprocess.run', side_effect=AssertionError('Unexpected Git command')), \
             patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected network')):
            runpy.run_path(str(ROOT / 'daily_arxiv.py'), run_name='__main__')
        for kind, before in original.items():
            after = json.loads(Path(config[f'json_{kind}_path']).read_text())
            self.assertEqual({k: set(v) for k, v in after.items()}, {k: set(v) for k, v in before.items()})
            self.assertTrue(Path(config[f'md_{kind}_path']).read_text())
        self.assertTrue(Path(config['md_gitpage_path']).read_text().startswith('---\nlayout: default\n---'))

    def test_daily_merges_without_losing_history_and_reuses_client(self):
        config = self.config()
        before = {kind: json.loads(Path(config[f'json_{kind}_path']).read_text()) for kind in ('readme', 'gitpage', 'wechat')}
        def result(topic, **kwargs):
            return {topic: {'2610.99999': '|date|new paper|author|paper|null|\n'}}, {topic: {'2610.99999': '- new paper\n'}}
        with patch.object(daily, 'get_daily_papers', side_effect=result) as fetch, patch.object(daily.arxiv, 'Client') as client:
            daily.demo(**config)
        client.assert_called_once_with(page_size=20, delay_seconds=3.0, num_retries=3)
        self.assertEqual(fetch.call_count, 11)
        for call in fetch.call_args_list:
            self.assertIs(call.kwargs['client'], client.return_value)
        for kind, history in before.items():
            after = json.loads(Path(config[f'json_{kind}_path']).read_text())
            for topic, rows in history.items():
                for key, value in rows.items():
                    self.assertEqual(after[topic][key], value)
            self.assertIn('2610.99999', after['LLM-for-Embodied-AI'])
            self.assertIn('## LLM-Technical-Reports', Path(config[f'md_{kind}_path']).read_text())

    def test_partial_fetch_failure_does_not_rewrite_history(self):
        config = self.config()
        files = [Path(config[f'json_{k}_path']) for k in ('readme', 'gitpage', 'wechat')]
        before = [p.read_bytes() for p in files]
        with patch.object(daily, 'get_daily_papers', side_effect=[({'a': {}}, {'a': {}}), requests.ConnectionError('offline')]):
            with self.assertRaises(requests.ConnectionError):
                daily.demo(**config)
        self.assertEqual([p.read_bytes() for p in files], before)


if __name__ == '__main__':
    unittest.main()
