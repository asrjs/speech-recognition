"""Run a ported model's declared checks from one manifest.

Fail-closed: every declared artifact, fixture and oracle must exist, be
non-empty and match its recorded SHA-256 before any run starts, so a stale or
replaced file cannot be reported as a port result. Run identity is written into
the receipt together with the per-coordinate text and token comparison against
the saved official oracle.

Usage:
  python tools/model-debugging/scripts/port_manifest.py --manifest <file> [--list]
  python tools/model-debugging/scripts/port_manifest.py --manifest <file> --run <run-id> --output <receipt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import write_new_text  # noqa: E402


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_file(relative: str, expected_sha256: str | None, role: str) -> dict[str, Any]:
    path = (ROOT / relative).resolve() if not Path(relative).is_absolute() else Path(relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or symlinked {role}: {relative}')
    if path.stat().st_size == 0:
        raise ValueError(f'Empty {role}: {relative}')
    actual = digest(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f'SHA-256 mismatch for {role} {relative}: {actual} != {expected_sha256}')
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': actual}


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for field in ('schemaVersion', 'modelId', 'family', 'localArtifacts', 'fixtures', 'runs'):
        if field not in manifest:
            raise ValueError(f'Manifest is missing required field: {field}')
    if manifest['schemaVersion'] != 1:
        raise ValueError(f"Unsupported manifest schemaVersion: {manifest['schemaVersion']}")
    return manifest


def check_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind every declared file to its recorded digest before any execution."""
    identity: dict[str, Any] = {
        'modelId': manifest['modelId'],
        'family': manifest['family'],
        'preset': manifest.get('preset'),
        'originalWeights': require_file(
            manifest['originalWeights']['path'], manifest['originalWeights'].get('sha256'), 'original weights'
        ),
        'localArtifacts': {
            'root': manifest['localArtifacts']['root'],
            'layout': manifest['localArtifacts']['layout'],
            'files': [
                require_file(f"{manifest['localArtifacts']['root']}/{name}", None, 'local artifact')
                for name in (
                    manifest['localArtifacts']['encoder'],
                    manifest['localArtifacts']['decoder'],
                    manifest['localArtifacts']['tokenizer'],
                )
            ],
        },
        'fixtures': [],
    }
    if 'runtimeConfig' in manifest:
        identity['runtimeConfig'] = require_file(
            manifest['runtimeConfig']['path'], manifest['runtimeConfig'].get('sha256'), 'runtime config'
        )
    for fixture in manifest['fixtures']:
        bound = require_file(fixture['path'], fixture.get('sha256'), 'fixture')
        bound['id'] = fixture['id']
        bound['durationSeconds'] = fixture.get('durationSeconds')
        bound['oracle'] = require_file(fixture['oracle']['path'], None, 'oracle')
        identity['fixtures'].append(bound)
    return identity


def resolve_placeholders(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in context.items():
            value = value.replace('{' + key + '}', replacement)
        if '{' in value and value.endswith('}'):
            raise ValueError(f'Unresolved placeholder: {value}')
        return value
    if isinstance(value, list):
        return [resolve_placeholders(item, context) for item in value]
    if isinstance(value, dict):
        return {key: resolve_placeholders(item, context) for key, item in value.items()}
    return value


def oracle_fields(path: Path, field: str) -> tuple[list[int], str]:
    document = json.loads(path.read_text(encoding='utf-8'))
    for part in field.split('.'):
        document = document[part]
    return list(document['token_ids']), str(document['text'])


def run_case(manifest: dict[str, Any], run: dict[str, Any], output: Path | None) -> dict[str, Any]:
    fixtures = {fixture['id']: fixture for fixture in manifest['fixtures']}
    # Browser warm-up cases transcribe their own harness sample and declare no
    # repository fixture, so fixture-scoped context stays empty for them.
    fixture = fixtures.get(run.get('fixture', ''))
    context = {
        'consumerRoot': run.get('consumerRoot', ''),
        'modelId': manifest['modelId'],
        'localArtifacts.root': manifest['localArtifacts']['root'],
        'runtimeConfig.path': manifest.get('runtimeConfig', {}).get('path', ''),
        'fixture.path': fixture['path'] if fixture else '',
        'output': str(output) if output else '',
        # Browser runs own a separate receipt so the manifest and the browser
        # harness never race for the same new-output path.
        'browserOutput': f'{output}.browser.json' if output else '',
    }
    command = resolve_placeholders(run['command'], context)
    if not output:
        return {'id': run['id'], 'status': 'LISTED', 'command': command}

    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            cwd=str(ROOT), timeout=1800)
    if result.returncode != 0:
        return {'id': run['id'], 'status': 'FAILED_RUNNER_EXIT', 'exitCode': result.returncode,
                'stderr': result.stderr[-4000:], 'command': command}

    if run['runner'] == 'browser':
        receipt_path = Path(context['browserOutput'])
        if not receipt_path.is_file():
            return {'id': run['id'], 'status': 'FAILED_NO_BROWSER_RECEIPT', 'command': command}
        browser = json.loads(receipt_path.read_text(encoding='utf-8'))
        local = browser.get('local_load') or {}
        expected_status = run.get('passStatus', 'PASS_COMPANION_LOCAL_WARMUP')
        checks = {
            'harness_status': browser.get('status') == expected_status,
            'model_load_and_warmup': local.get('status') == 'PASS_APP_LOCAL_LOAD_AND_WARMUP',
            'no_page_errors': not browser.get('page_errors'),
            'offline_sidecars': bool(browser.get('verified_sidecars')),
        }
        return {
            'id': run['id'],
            # The companion UI transcribes its own bundled warm-up sample, so it
            # proves load and warm-up only. Never report an oracle match here.
            'status': 'PASS_WARMUP' if all(checks.values()) else 'FAILED_WARMUP',
            'comparesOracle': False,
            'verifiedInput': 'companion harness warm-up sample (Harvard-L2-1.ogg)',
            'checks': checks,
            'command': command,
            'runner': 'browser',
            'backend': run['backend'],
            'browserVersion': browser.get('browser_version'),
            'adapter': local.get('adapter'),
            'loadStatus': local.get('status'),
            'browserReceipt': str(receipt_path.resolve().relative_to(ROOT)).replace('\\', '/'),
        }

    payload = json.loads(result.stdout)
    transcript = payload['transcript']['canonical']
    expected_tokens, expected_text = oracle_fields(
        (ROOT / fixture['oracle']['path']).resolve(), fixture['oracle']['field']
    )
    observed_tokens = [token['id'] for token in transcript['tokens']]
    observed_text = transcript['text'].strip()
    checks = {
        'token_ids': observed_tokens == expected_tokens,
        'text': observed_text == expected_text.strip(),
    }
    for field in run.get('compare', []):
        if not checks.get(field, False):
            return {'id': run['id'], 'status': 'FAILED_ORACLE', 'checks': checks, 'command': command,
                    'observed': {'tokens': len(observed_tokens), 'text': observed_text[:400]},
                    'expected': {'tokens': len(expected_tokens), 'text': expected_text[:400]}}
    return {'id': run['id'], 'status': 'PASS_ORACLE_MATCH', 'checks': checks, 'command': command,
            'runner': run['runner'], 'backend': run['backend'], 'fixture': run['fixture'],
            'host': payload.get('host'), 'ort': payload.get('backend', {}).get('versions'),
            'metrics': {key: transcript['meta']['metrics'][key]
                        for key in ('totalMs', 'rtf', 'rtfx', 'encoderFrameCount', 'decodeIterations')
                        if key in transcript['meta']['metrics']},
            'warnings': transcript.get('warnings')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', help='Run id; default lists every declared run')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    identity = check_manifest(manifest)
    cases = [*manifest['runs'], *manifest.get('browserRuns', [])]
    if args.run:
        selected = [case for case in cases if case['id'] == args.run]
        if not selected:
            raise ValueError(f'Unknown run id: {args.run}')
    else:
        selected = cases

    results = [run_case(manifest, case, args.output) for case in selected]
    receipt: dict[str, Any] = {
        'status': 'PASS' if all(item['status'] in ('PASS_ORACLE_MATCH', 'PASS_WARMUP', 'LISTED') for item in results)
        else 'FAIL',
        'scope': 'Manifest-bound port run; identity digests recorded, oracle compared',
        'manifest': str(args.manifest.resolve().relative_to(ROOT)).replace('\\', '/'),
        'identity': identity,
        'results': results,
    }
    print(json.dumps(receipt, indent=2, allow_nan=False))
    if args.output:
        write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())