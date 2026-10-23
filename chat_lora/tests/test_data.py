import unittest

from chat_lora.data import DialoguePair, PadCollator, encode_example, extract_pairs, pairs_from_messages


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "SYSTEM\\n" + "\\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\\nASSISTANT: "

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        # A deterministic toy tokenizer: one token per whitespace-separated word.
        ids = [10 + (abs(hash(word)) % 50) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class DataTests(unittest.TestCase):
    def test_extract_pairs(self):
        dialog = ["hello there", "hi how are you", "i am fine", "that is good"]
        pairs = extract_pairs(dialog)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].user, "hello there")
        self.assertEqual(pairs[0].assistant, "hi how are you")
        self.assertEqual(pairs[1].user, "i am fine")

    def test_pairs_from_messages(self):
        dataset = [
            {"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you"},
                {"role": "assistant", "content": "i am fine"},
            ]}
        ]
        pairs = pairs_from_messages(dataset)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].assistant, "hi there")
        self.assertEqual(pairs[1].user, "how are you")

    def test_encode_masks_prompt(self):
        tokenizer = FakeTokenizer()
        item = encode_example(tokenizer, DialoguePair("hello", "hi there"), max_len=256)
        self.assertIsNotNone(item)
        prompt_len = sum(1 for label in item["labels"] if label == -100)
        self.assertGreater(prompt_len, 0)
        self.assertEqual(len(item["input_ids"]), len(item["labels"]))

    def test_collator(self):
        collator = PadCollator(0)
        batch = collator([
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ])
        self.assertEqual(batch["input_ids"].shape, (2, 2))
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
