import unittest

import torch

from concept_chat.data import BOS, EOS, PAD, SPECIALS, build_corpus, decode, encode
from concept_chat.model import ConceptSeq2Seq, count_parameters
from concept_chat.train import make_batch, token_f1


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus(1234)
        cls.concept_to_id = {name: i for i, name in enumerate(cls.corpus.concepts)}

    def test_specials(self):
        self.assertEqual(self.corpus.itos[:4], SPECIALS)
        self.assertEqual(self.corpus.vocab[PAD], 0)
        self.assertEqual(self.corpus.vocab[BOS], 2)
        self.assertEqual(self.corpus.vocab[EOS], 3)

    def test_splits_nonempty(self):
        self.assertGreater(len(self.corpus.train), 500)
        self.assertGreater(len(self.corpus.val), 100)
        for name, examples in self.corpus.tests.items():
            self.assertGreater(len(examples), 20, name)

    def test_canonical_chat_prompts_are_trained(self):
        users = {example.user for example in self.corpus.train}
        for prompt in [
            "hi", "hello", "what is your name", "i am sad", "i want an apple",
            "what sound does a cow make", "good night",
            "the ball is red what color is the ball",
            "the cat is under the table where is the cat",
            "i see three apples how many apples",
        ]:
            self.assertIn(prompt, users, prompt)

    def test_no_train_leak_into_test(self):
        train = {(e.user, e.reply) for e in self.corpus.train}
        for name, examples in self.corpus.tests.items():
            for example in examples:
                self.assertNotIn((example.user, example.reply), train, name)

    def test_encode_decode_roundtrip(self):
        text = "what color is the ball"
        ids = encode(text, self.corpus.vocab, 20)
        self.assertEqual(decode(ids, self.corpus.itos), text)

    def test_make_batch_shapes(self):
        batch = make_batch(self.corpus.train[:4], self.corpus, self.concept_to_id, torch.device("cpu"))
        src, dec_in, dec_out, concepts = batch
        self.assertEqual(src.shape[0], 4)
        self.assertEqual(src.shape[1], self.corpus.max_src_len)
        self.assertEqual(dec_in.shape, dec_out.shape)
        self.assertEqual(concepts.shape, (4,))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus(1234)
        cls.concept_to_id = {name: i for i, name in enumerate(cls.corpus.concepts)}
        cls.model = ConceptSeq2Seq(len(cls.corpus.vocab), len(cls.corpus.concepts),
                                   emb=16, hidden=32, dropout=0.0, use_concept=True)

    def test_forward_shapes(self):
        src, dec_in, dec_out, concepts = make_batch(
            self.corpus.train[:3], self.corpus, self.concept_to_id, torch.device("cpu")
        )
        logits, concept_logits = self.model(src, dec_in, concepts)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits.shape[2], len(self.corpus.vocab))
        self.assertEqual(concept_logits.shape, (3, len(self.corpus.concepts)))

    def test_log_probs_and_nll(self):
        src, dec_in, dec_out, concepts = make_batch(
            self.corpus.train[:3], self.corpus, self.concept_to_id, torch.device("cpu")
        )
        log_probs, _ = self.model(src, dec_in, concepts)
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))
        loss = torch.nn.functional.nll_loss(
            log_probs.reshape(-1, log_probs.shape[-1]), dec_out.reshape(-1), ignore_index=0
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_generate(self):
        src, _, _, concepts = make_batch(
            self.corpus.train[:2], self.corpus, self.concept_to_id, torch.device("cpu")
        )
        generated, predicted = self.model.generate(src, 5)
        self.assertEqual(len(generated), 2)
        self.assertEqual(predicted.shape, (2,))

    def test_baseline_has_no_concept_head(self):
        model = ConceptSeq2Seq(len(self.corpus.vocab), len(self.corpus.concepts),
                               emb=16, hidden=32, dropout=0.0, use_concept=False)
        src, dec_in, _, _ = make_batch(
            self.corpus.train[:2], self.corpus, self.concept_to_id, torch.device("cpu")
        )
        logits, concept_logits = model(src, dec_in)
        self.assertIsNone(concept_logits)
        self.assertEqual(logits.shape[0], 2)

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("a b c", "a b c"), 1.0)
        self.assertGreater(token_f1("a b c", "a b d"), 0.0)
        self.assertEqual(token_f1("a", "b"), 0.0)


if __name__ == "__main__":
    unittest.main()
