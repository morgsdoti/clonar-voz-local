import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import clonar
import transcrever


class DeviceTests(unittest.TestCase):
    def backend(self, cuda=False, mps=False):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        )

    def test_cpu(self):
        with patch.dict(sys.modules, torch=self.backend()):
            self.assertEqual(clonar.select_device('auto'), 'cpu')

    def test_cuda_preferred(self):
        with patch.dict(sys.modules, torch=self.backend(cuda=True, mps=True)):
            self.assertEqual(clonar.select_device('auto'), 'cuda:0')

    def test_mps(self):
        with patch.dict(sys.modules, torch=self.backend(mps=True)):
            self.assertEqual(clonar.select_device('auto'), 'mps')

    def test_explicit_cpu(self):
        with patch.dict(sys.modules, torch=self.backend(cuda=True)):
            self.assertEqual(clonar.select_device('cpu'), 'cpu')

    def test_unavailable_gpu(self):
        with patch.dict(sys.modules, torch=self.backend()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(clonar.select_device('cuda'), 'cpu')


class CLITests(unittest.TestCase):
    def invoke(self, directory, **kwargs):
        out = Path(directory) / 'out.wav'
        args = ['clonar.py', '--ref', str(Path(directory) / 'ref.wav'),
                '--ref-text', 'uma referência genérica', '--texto', 'teste local.',
                '--out', str(out), '--consentimento']
        with patch.object(sys, 'argv', args), patch.object(clonar, 'load_reference', return_value=(np.ones(100), 24000)), \
                patch.object(clonar, 'select_device', return_value=kwargs.get('device', 'cpu')), \
                patch.object(clonar, 'synthesize', **kwargs['mock']) as synth, \
                patch.object(clonar, 'release_accelerator') as release, \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            status = clonar.main()
        return status, out, synth, release

    def test_mocked_cli_saves_pcm16(self):
        with tempfile.TemporaryDirectory() as directory:
            status, out, synth, _ = self.invoke(directory, mock={'return_value': (np.ones(24000)*.1, 24000)})
            self.assertEqual(status, 0)
            self.assertEqual(sf.info(out).subtype, 'PCM_16')
            self.assertEqual(sf.info(out).samplerate, 24000)
            self.assertEqual(synth.call_count, 1)

    def test_backend_fallback_restarts_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            status, out, synth, release = self.invoke(
                directory, device='cuda:0', mock={'side_effect': [RuntimeError('CUDA out of memory'), (np.ones(24000)*.1, 24000)]})
            self.assertEqual(status, 0)
            self.assertTrue(out.exists())
            self.assertEqual([c.args[-1] for c in synth.call_args_list], ['cuda:0', 'cpu'])
            release.assert_called_once()

    def test_network_error_not_retried_as_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            status, out, synth, release = self.invoke(
                directory, device='cuda:0', mock={'side_effect': RuntimeError('download denied')})
            self.assertEqual(status, 1)
            self.assertFalse(out.exists())
            self.assertEqual(synth.call_count, 1)
            release.assert_not_called()

    def test_consent_required(self):
        with patch.object(sys, 'argv', ['clonar.py', '--ref', 'ref.wav', '--texto', 'teste']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            clonar.main()
        self.assertEqual(caught.exception.code, 2)

    def test_transcript_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'ref.txt'
            out.write_text('original', encoding='utf-8')
            with patch.object(sys, 'argv', ['transcrever.py', '--ref', 'ref.wav', '--out', str(out)]), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(transcrever.main(), 1)
            self.assertEqual(out.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
