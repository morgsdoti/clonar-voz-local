import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_utils import as_mono, check_output, load_reference, split_text


class AudioTests(unittest.TestCase):
    def test_sentence_boundaries(self):
        self.assertEqual(split_text('Olá mundo. Tudo bem?'), ['Olá mundo.', 'Tudo bem?'])

    def test_long_text_preserves_words(self):
        text = 'uma frase longa sem ponto ' * 100
        chunks = split_text(text, 80)
        self.assertEqual(' '.join(chunks), ' '.join(text.split()))
        self.assertTrue(all(len(c) <= 80 for c in chunks))

    def test_long_token(self):
        text = 'x' * 501
        chunks = split_text(text, 100)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            split_text(' \n\t ')

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            split_text('Olá', 0)

    def test_stereo_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.wav'
            sf.write(path, np.full((24000, 2), .1), 24000)
            audio, rate = load_reference(path)
            self.assertEqual(rate, 24000)
            self.assertEqual(audio.shape, (24000,))

    def test_reference_too_long(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.wav'
            sf.write(path, np.full(21000, .1), 1000)
            with self.assertRaises(ValueError):
                load_reference(path)

    def test_silent_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.wav'
            sf.write(path, np.zeros(24000), 24000)
            with self.assertRaises(ValueError):
                load_reference(path)

    def test_invalid_generation(self):
        for audio in [np.array([]), np.array([0., np.nan]), np.ones((2, 8))]:
            with self.assertRaises(ValueError):
                as_mono(audio)

    def test_clips_generation(self):
        self.assertTrue(np.array_equal(as_mono(np.array([-2., 2.])), [-1., 1.]))

    def test_protect_reference(self):
        with self.assertRaises(ValueError):
            check_output(Path('same.wav'), [Path('same.wav')])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.wav'
            path.touch()
            with self.assertRaises(FileExistsError):
                check_output(path, [])

    def test_wav_extension(self):
        with self.assertRaises(ValueError):
            check_output(Path('output.mp3'), [])


if __name__ == '__main__':
    unittest.main()
