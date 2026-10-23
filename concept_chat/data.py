"""Synthetic baby-English dialogue corpus with concept and grammar structure.

Each concept has several user/reply template pairs and a set of slot fillers.
Splits are made at both the template level and the slot-combination level, so
the test suite measures paraphrase generalization and novel-combination
generalization separately.

This is deliberately a closed-world synthetic corpus, not open-domain English.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Iterable

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


@dataclass(frozen=True)
class Template:
    template_id: str
    user: str
    reply: str


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    templates: list[Template]
    slots: list[dict[str, str]]


@dataclass(frozen=True)
class Example:
    concept: str
    user: str
    reply: str
    template_id: str
    slot_key: str


@dataclass
class Corpus:
    train: list[Example]
    val: list[Example]
    tests: dict[str, list[Example]]
    concepts: list[str]
    vocab: dict[str, int]
    itos: list[str]
    max_src_len: int
    max_tgt_len: int
    stats: dict = field(default_factory=dict)


def _t(concept: str, index: int, user: str, reply: str) -> Template:
    return Template(f"{concept}_{index}", user, reply)


def _slots(**kwargs) -> dict[str, str]:
    return kwargs


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def concept_specs() -> list[ConceptSpec]:
    specs: list[ConceptSpec] = []

    specs.append(ConceptSpec("greeting", [
        _t("greeting", 0, "hi", "hello how are you"),
        _t("greeting", 1, "hello", "hi i am happy to see you"),
        _t("greeting", 2, "hey", "hey do you want to play"),
        _t("greeting", 3, "good morning", "good morning did you sleep well"),
        _t("greeting", 4, "good evening", "good evening how was your day"),
        _t("greeting", 5, "hi pip", "hello i am pip how are you"),
        _t("greeting", 6, "hello friend", "hello friend what do you want to do"),
        _t("greeting", 7, "hi there", "hi there i am here with you"),
        _t("greeting", 8, "hello there", "hello how are you"),
        _t("greeting", 9, "good afternoon", "good afternoon i am happy to see you"),
        _t("greeting", 10, "howdy", "howdy friend"),
        _t("greeting", 11, "morning", "good morning did you sleep well"),
    ], [_slots()]))

    specs.append(ConceptSpec("name", [
        _t("name", 0, "what is your name", "my name is pip"),
        _t("name", 1, "who are you", "i am pip a little learner"),
        _t("name", 2, "what should i call you", "you can call me pip"),
        _t("name", 3, "do you have a name", "yes my name is pip"),
        _t("name", 4, "tell me your name", "my name is pip"),
        _t("name", 5, "are you pip", "yes i am pip"),
        _t("name", 6, "what are you", "i am a small talking friend"),
        _t("name", 7, "can you tell me your name", "of course i am pip"),
        _t("name", 8, "what is your name pip", "my name is pip"),
        _t("name", 9, "tell me who you are", "i am pip a little helper"),
        _t("name", 10, "what do people call you", "i am called pip"),
        _t("name", 11, "your name please", "my name is pip"),
    ], [_slots()]))

    feelings = [
        _slots(feeling="happy", comfort="i am glad you are happy"),
        _slots(feeling="sad", comfort="i am sorry you are sad do you want a hug"),
        _slots(feeling="tired", comfort="do you want to rest"),
        _slots(feeling="hungry", comfort="do you want an apple"),
        _slots(feeling="thirsty", comfort="do you want some water"),
        _slots(feeling="scared", comfort="i am here with you"),
        _slots(feeling="angry", comfort="take a deep breath i am here"),
        _slots(feeling="sleepy", comfort="do you want to sleep"),
        _slots(feeling="excited", comfort="that is wonderful what made you excited"),
        _slots(feeling="lonely", comfort="i am here with you we can talk"),
    ]
    specs.append(ConceptSpec("feelings", [
        _t("feelings", 0, "i am {feeling}", "{comfort}"),
        _t("feelings", 1, "i feel {feeling}", "{comfort}"),
        _t("feelings", 2, "i am feeling {feeling}", "{comfort}"),
        _t("feelings", 3, "i feel very {feeling}", "{comfort}"),
        _t("feelings", 4, "i am {feeling} today", "{comfort}"),
        _t("feelings", 5, "i am so {feeling}", "{comfort}"),
        _t("feelings", 6, "right now i am {feeling}", "{comfort}"),
        _t("feelings", 7, "i feel a little {feeling}", "{comfort}"),
    ], feelings))

    specs.append(ConceptSpec("politeness", [
        _t("politeness", 0, "thank you", "you are welcome"),
        _t("politeness", 1, "thanks", "you are welcome i am happy to help"),
        _t("politeness", 2, "please help me", "of course i will help you"),
        _t("politeness", 3, "sorry", "that is okay we can try again"),
        _t("politeness", 4, "excuse me", "yes i am listening"),
        _t("politeness", 5, "you are welcome", "thank you for being polite"),
        _t("politeness", 6, "please", "you are very polite thank you"),
        _t("politeness", 7, "i am sorry", "do not worry it is okay"),
        _t("politeness", 8, "many thanks", "you are welcome"),
        _t("politeness", 9, "please and thank you", "you are very polite"),
        _t("politeness", 10, "i apologize", "that is okay"),
        _t("politeness", 11, "pardon me", "yes i am listening"),
    ], [_slots()]))

    colors = [
        _slots(object=obj, color=color, article=_article(color))
        for obj in ["ball", "car", "cup", "hat", "book", "flower", "box", "chair", "bag", "shoe"]
        for color in ["red", "blue", "green", "yellow", "black", "white", "pink", "orange"]
    ]
    specs.append(ConceptSpec("colors", [
        _t("colors", 0, "the {object} is {color} what color is the {object}", "the {object} is {color}"),
        _t("colors", 1, "i see {article} {color} {object} what color is the {object}", "it is {color}"),
        _t("colors", 2, "look at the {article} {color} {object} tell me the color of the {object}", "the {object} looks {color}"),
        _t("colors", 3, "the {object} is {color} do you know its color", "i see {article} {color} {object}"),
        _t("colors", 4, "here is {article} {color} {object} which color is the {object}", "the {object} is {color}"),
        _t("colors", 5, "this {object} is {color} what color do you see", "it is {article} {color} {object}"),
        _t("colors", 6, "the {object} is {color} can you tell me the color", "i see {color}"),
        _t("colors", 7, "i have {article} {color} {object} what color is it", "the {object} is {color}"),
    ], colors))

    animals = [
        _slots(animal="cow", sound="moo"), _slots(animal="cat", sound="meow"),
        _slots(animal="dog", sound="woof"), _slots(animal="duck", sound="quack"),
        _slots(animal="sheep", sound="baa"), _slots(animal="bird", sound="tweet"),
        _slots(animal="pig", sound="oink"), _slots(animal="horse", sound="neigh"),
        _slots(animal="frog", sound="ribbit"), _slots(animal="lion", sound="roar"),
        _slots(animal="cow", sound="moo"), _slots(animal="owl", sound="hoot"),
    ]
    # Deduplicate animal slot dictionaries while preserving order.
    animals = list({(s["animal"], s["sound"]): s for s in animals}.values())
    animals = [
        _slots(animal=s["animal"], sound=s["sound"], article=_article(s["animal"]))
        for s in animals
    ]
    specs.append(ConceptSpec("animals", [
        _t("animals", 0, "what sound does {article} {animal} make", "{article} {animal} says {sound}"),
        _t("animals", 1, "what does {article} {animal} say", "the {animal} says {sound}"),
        _t("animals", 2, "what noise does a {animal} make", "{animal} says {sound}"),
        _t("animals", 3, "can you tell me the sound of {article} {animal}", "it says {sound}"),
        _t("animals", 4, "how does {article} {animal} talk", "{article} {animal} says {sound}"),
        _t("animals", 5, "what does the {animal} sound like", "the {animal} says {sound}"),
        _t("animals", 6, "tell me the sound {article} {animal} makes", "{animal} says {sound}"),
        _t("animals", 7, "do you know what {article} {animal} says", "yes it says {sound}"),
    ], animals))

    counting = []
    for number, word in [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]:
        for singular, plural in [
            ("apple", "apples"), ("cat", "cats"), ("ball", "balls"), ("book", "books"),
            ("cup", "cups"), ("cookie", "cookies"), ("flower", "flowers"), ("bird", "birds"),
        ]:
            noun = singular if number == 1 else plural
            there = "there is" if number == 1 else "there are"
            here = "here is" if number == 1 else "here are"
            counting.append(_slots(
                number=word, plural=plural, noun=noun, there=there, here=here,
                count_phrase=f"{word} {noun}",
            ))
    specs.append(ConceptSpec("counting", [
        _t("counting", 0, "i see {count_phrase} how many {plural}", "{there} {number} {noun}"),
        _t("counting", 1, "look {here} {count_phrase} count the {plural}", "{there} {number} {noun}"),
        _t("counting", 2, "look at {count_phrase} how many {plural} do you see", "i see {number} {noun}"),
        _t("counting", 3, "{here} {count_phrase} tell me how many {plural} there are", "{there} {number} {noun}"),
        _t("counting", 4, "i have {count_phrase} can you count the {plural}", "yes {there} {number} {noun}"),
        _t("counting", 5, "the number of {plural} is {number} what is the number", "the number is {number}"),
        _t("counting", 6, "i see {count_phrase} how many {plural} are here", "{there} {number} {noun} here"),
        _t("counting", 7, "count with me {count_phrase} do you see the {plural} how many", "i see {number} {noun}"),
    ], counting))

    prepositions = [
        _slots(object=obj, prep=prep, place=place)
        for obj in ["cat", "ball", "book", "cup", "toy", "bag", "hat", "box"]
        for place in ["table", "box", "bed", "chair", "floor", "shelf", "rug", "bag"]
        for prep in ["on", "in", "under"]
    ]
    specs.append(ConceptSpec("prepositions", [
        _t("prepositions", 0, "the {object} is {prep} the {place} where is the {object}", "the {object} is {prep} the {place}"),
        _t("prepositions", 1, "i see the {object} {prep} the {place} where is it", "it is {prep} the {place}"),
        _t("prepositions", 2, "the {object} is {prep} the {place} can you tell me where it is", "i see it {prep} the {place}"),
        _t("prepositions", 3, "look the {object} is {prep} the {place} where do you see it", "the {object} is {prep} the {place}"),
        _t("prepositions", 4, "i put the {object} {prep} the {place} where did it go", "it is {prep} the {place} now"),
        _t("prepositions", 5, "the {object} is {prep} the {place} where is it now", "yes the {object} is {prep} the {place}"),
        _t("prepositions", 6, "the {object} is {prep} the {place} tell me where the {object} is", "the {object} is {prep} the {place}"),
        _t("prepositions", 7, "the {object} is {prep} the {place} do you know where it is", "yes it is {prep} the {place}"),
    ], prepositions))

    wants = [
        _slots(item="apple", article="an"), _slots(item="banana", article="a"),
        _slots(item="cookie", article="a"), _slots(item="orange", article="an"),
        _slots(item="egg", article="an"), _slots(item="book", article="a"),
        _slots(item="ball", article="a"), _slots(item="cup", article="a"),
        _slots(item="sandwich", article="a"), _slots(item="umbrella", article="an"),
        _slots(item="ice cream", article="an"), _slots(item="toy", article="a"),
    ]
    specs.append(ConceptSpec("wants", [
        _t("wants", 0, "i want {article} {item}", "here is {article} {item}"),
        _t("wants", 1, "can i have {article} {item}", "yes here is {article} {item}"),
        _t("wants", 2, "i would like {article} {item}", "of course here is {article} {item}"),
        _t("wants", 3, "please give me {article} {item}", "you can have {article} {item}"),
        _t("wants", 4, "may i have {article} {item}", "yes here is {article} {item}"),
        _t("wants", 5, "i am hungry for {article} {item}", "here is {article} {item}"),
        _t("wants", 6, "i want to eat {article} {item}", "here is {article} {item}"),
        _t("wants", 7, "can you give me {article} {item}", "yes here is {article} {item}"),
    ], wants))

    likes = [_slots(thing=t) for t in [
        "cats", "dogs", "apples", "books", "balls", "flowers", "birds", "cookies",
        "cars", "stories", "games", "songs",
    ]]
    specs.append(ConceptSpec("likes", [
        _t("likes", 0, "i like {thing}", "i like {thing} too"),
        _t("likes", 1, "i love {thing}", "i love {thing} too"),
        _t("likes", 2, "do you like {thing}", "yes i like {thing}"),
        _t("likes", 3, "i really like {thing}", "{thing} are nice"),
        _t("likes", 4, "i enjoy {thing}", "i enjoy {thing} too"),
        _t("likes", 5, "are {thing} nice", "yes {thing} are nice"),
        _t("likes", 6, "i like {thing} a lot", "i do too"),
        _t("likes", 7, "do you love {thing}", "yes i love {thing}"),
    ], likes))

    specs.append(ConceptSpec("day", [
        _t("day", 0, "good night", "good night sleep well"),
        _t("day", 1, "good afternoon", "good afternoon how are you"),
        _t("day", 2, "good evening", "good evening i am happy to see you"),
        _t("day", 3, "bye", "bye bye see you soon"),
        _t("day", 4, "see you", "see you later friend"),
        _t("day", 5, "good night pip", "good night friend sleep well"),
        _t("day", 6, "sleep well", "thank you you too"),
        _t("day", 7, "good morning pip", "good morning i missed you"),
        _t("day", 8, "good night friend", "good night sleep well"),
        _t("day", 9, "bye bye", "bye bye see you soon"),
        _t("day", 10, "good afternoon pip", "good afternoon"),
        _t("day", 11, "see you tomorrow", "see you tomorrow friend"),
    ], [_slots()]))

    specs.append(ConceptSpec("help", [
        _t("help", 0, "can you help me", "yes i can help you"),
        _t("help", 1, "i need help", "i will try to help"),
        _t("help", 2, "help me please", "tell me what you need"),
        _t("help", 3, "can you help", "yes what do you need"),
        _t("help", 4, "i am stuck", "i can help you think"),
        _t("help", 5, "what can you do", "i can talk and answer simple questions"),
        _t("help", 6, "can you teach me", "yes i can teach you simple words"),
        _t("help", 7, "i do not understand", "that is okay i will say it again"),
        _t("help", 8, "please help", "i will help you"),
        _t("help", 9, "can you assist me", "yes i can help"),
        _t("help", 10, "i need a hand", "i am here to help"),
        _t("help", 11, "help", "what do you need"),
    ], [_slots()]))

    specs.append(ConceptSpec("smalltalk", [
        _t("smalltalk", 0, "how are you", "i am fine thank you"),
        _t("smalltalk", 1, "what are you doing", "i am talking with you"),
        _t("smalltalk", 2, "do you want to play", "yes i want to play"),
        _t("smalltalk", 3, "are you happy", "yes i am happy"),
        _t("smalltalk", 4, "what do you like", "i like stories and games"),
        _t("smalltalk", 5, "tell me something", "i am a little learner and i like you"),
        _t("smalltalk", 6, "are you my friend", "yes i am your friend"),
        _t("smalltalk", 7, "do you love me", "yes i love you friend"),
        _t("smalltalk", 8, "how are you today", "i am fine thank you"),
        _t("smalltalk", 9, "what is up", "i am here with you"),
        _t("smalltalk", 10, "do you want to talk", "yes i want to talk"),
        _t("smalltalk", 11, "are you okay", "yes i am okay"),
    ], [_slots()]))

    return specs


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def _split_indices(count: int, rng: random.Random, train_frac: float = 0.6,
                   val_frac: float = 0.2) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(count))
    rng.shuffle(indices)
    n_train = max(1, int(round(count * train_frac)))
    n_val = max(1, int(round(count * val_frac))) if count - n_train > 1 else 0
    if n_train + n_val >= count:
        n_val = max(0, count - n_train - 1)
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    if not test:
        test = [train.pop()]
    return train, val, test


def _split_templates(count: int, rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    """Keep the canonical first two prompts in train, hold out paraphrases.

    The canonical prompts are what the interactive chat examples use.  Holding
    them out makes chat look broken even when the slot/composition test works.
    """
    forced = [index for index in (0, 1) if index < count]
    rest = [index for index in range(count) if index not in forced]
    rng.shuffle(rest)
    n_train = max(len(forced), int(round(count * 0.6)))
    train = forced + rest[: max(0, n_train - len(forced))]
    remaining = rest[max(0, n_train - len(forced)) :]
    n_val = max(1, int(round(count * 0.2))) if len(remaining) > 1 else 0
    val = remaining[:n_val]
    test = remaining[n_val:]
    if not test and val:
        test = [val.pop()]
    return train, val, test


def _split_slots(count: int, rng: random.Random, forced: list[int]) -> tuple[list[int], list[int], list[int]]:
    rest = [index for index in range(count) if index not in forced]
    rng.shuffle(rest)
    n_train = max(len(forced), int(round(count * 0.7)))
    train = forced + rest[: max(0, n_train - len(forced))]
    remaining = rest[max(0, n_train - len(forced)) :]
    n_val = max(1, int(round(count * 0.15))) if len(remaining) > 1 else 0
    val = remaining[:n_val]
    test = remaining[n_val:]
    if not test and val:
        test = [val.pop()]
    return train, val, test


CANONICAL_SLOTS = {
    "colors": [
        {"object": "ball", "color": "red"},
        {"object": "ball", "color": "blue"},
    ],
    "animals": [
        {"animal": "cow", "sound": "moo"},
        {"animal": "cat", "sound": "meow"},
        {"animal": "dog", "sound": "woof"},
    ],
    "counting": [
        {"number": "three", "plural": "apples"},
        {"number": "one", "plural": "apples"},
        {"number": "two", "plural": "apples"},
    ],
    "prepositions": [
        {"object": "cat", "prep": "under", "place": "table"},
        {"object": "cat", "prep": "on", "place": "table"},
        {"object": "ball", "prep": "on", "place": "table"},
    ],
    "wants": [
        {"item": "apple", "article": "an"},
        {"item": "banana", "article": "a"},
    ],
    "likes": [{"thing": "cats"}, {"thing": "dogs"}],
    "feelings": [
        {"feeling": "sad", "comfort": "i am sorry you are sad do you want a hug"},
        {"feeling": "happy", "comfort": "i am glad you are happy"},
    ],
}


def _make_examples(spec: ConceptSpec, template_indices: Iterable[int],
                   slot_indices: Iterable[int]) -> list[Example]:
    out: list[Example] = []
    seen: set[tuple[str, str]] = set()
    for ti in template_indices:
        template = spec.templates[ti]
        for si in slot_indices:
            slots = spec.slots[si]
            user = normalize(template.user.format(**slots))
            reply = normalize(template.reply.format(**slots))
            key = (user, reply)
            if key in seen:
                continue
            seen.add(key)
            slot_key = "|".join(f"{key}={value}" for key, value in sorted(slots.items()))
            out.append(Example(spec.name, user, reply, template.template_id, slot_key))
    return out


def build_corpus(seed: int = 1234) -> Corpus:
    specs = concept_specs()
    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    tests: dict[str, list[Example]] = {
        "test_seen_template_unseen_slots": [],
        "test_unseen_templates_seen_slots": [],
        "test_unseen_both": [],
    }
    stats: dict = {"concepts": {}, "splits": {}}

    for spec in specs:
        n_t = len(spec.templates)
        n_s = len(spec.slots)
        t_train, t_val, t_test = _split_templates(n_t, rng)
        forced_slots = []
        for wanted in CANONICAL_SLOTS.get(spec.name, []):
            for index, slot in enumerate(spec.slots):
                if all(slot.get(key) == value for key, value in wanted.items()):
                    forced_slots.append(index)
                    break
        forced_slots = sorted(set(forced_slots))
        if n_s == 1:
            s_train, s_val, s_test = [0], [], []
        else:
            s_train, s_val, s_test = _split_slots(n_s, rng, forced_slots)

        c_train = _make_examples(spec, t_train, s_train)
        c_val = _make_examples(spec, t_val, s_train) + _make_examples(spec, t_train, s_val)
        c_seen_slots = _make_examples(spec, t_train, s_test)
        c_unseen_templates = _make_examples(spec, t_test, s_train)
        c_both = _make_examples(spec, t_test, s_test)

        # Deduplicate within / across evaluation sets and never leak a training
        # user/reply pair into evaluation.
        train_keys = {(e.user, e.reply) for e in c_train}
        def clean(items: list[Example], seen: set[tuple[str, str]]) -> list[Example]:
            out = []
            for item in items:
                key = (item.user, item.reply)
                if key in train_keys or key in seen:
                    continue
                seen.add(key)
                out.append(item)
            return out

        val_seen: set[tuple[str, str]] = set()
        c_val = clean(c_val, val_seen)
        test_seen: set[tuple[str, str]] = set()
        c_seen_slots = clean(c_seen_slots, test_seen)
        c_unseen_templates = clean(c_unseen_templates, test_seen)
        c_both = clean(c_both, test_seen)

        train.extend(c_train)
        val.extend(c_val)
        tests["test_seen_template_unseen_slots"].extend(c_seen_slots)
        tests["test_unseen_templates_seen_slots"].extend(c_unseen_templates)
        tests["test_unseen_both"].extend(c_both)
        stats["concepts"][spec.name] = {
            "templates": n_t, "slots": n_s,
            "train": len(c_train), "val": len(c_val),
            "test_seen_template_unseen_slots": len(c_seen_slots),
            "test_unseen_templates_seen_slots": len(c_unseen_templates),
            "test_unseen_both": len(c_both),
        }

    # Closed-world synthetic vocabulary: every word in the task specification is
    # known, but the model still has to learn the mapping. Unknown chat words are
    # mapped to <unk>.
    words = set()
    for example in train + val + tests["test_seen_template_unseen_slots"] + tests["test_unseen_templates_seen_slots"] + tests["test_unseen_both"]:
        words.update(example.user.split())
        words.update(example.reply.split())
    itos = SPECIALS + sorted(words)
    vocab = {token: index for index, token in enumerate(itos)}

    all_examples = (
        train + val
        + tests["test_seen_template_unseen_slots"]
        + tests["test_unseen_templates_seen_slots"]
        + tests["test_unseen_both"]
    )
    max_src = max((len(e.user.split()) for e in all_examples), default=1)
    max_tgt = max((len(e.reply.split()) + 1 for e in all_examples), default=2)
    stats["splits"] = {
        "train": len(train), "val": len(val),
        "test_seen_template_unseen_slots": len(tests["test_seen_template_unseen_slots"]),
        "test_unseen_templates_seen_slots": len(tests["test_unseen_templates_seen_slots"]),
        "test_unseen_both": len(tests["test_unseen_both"]),
        "vocab": len(vocab), "max_src_len": max_src, "max_tgt_len": max_tgt,
    }
    concepts = [spec.name for spec in specs]
    return Corpus(train, val, tests, concepts, vocab, itos, max_src, max_tgt, stats)


def encode(text: str, vocab: dict[str, int], max_len: int, add_bos: bool = False,
           add_eos: bool = False) -> list[int]:
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize(text)]
    if add_bos:
        ids = [vocab[BOS]] + ids
    if add_eos:
        ids = ids + [vocab[EOS]]
    return ids[:max_len] if max_len > 0 else ids


def decode(ids: Iterable[int], itos: list[str]) -> str:
    words = []
    for token_id in ids:
        if token_id in (0, 1, 2, 3):
            continue
        words.append(itos[token_id])
    return " ".join(words)


def pad_batch(sequences: list[list[int]], pad_id: int, max_len: int | None = None) -> list[list[int]]:
    width = max_len or max((len(seq) for seq in sequences), default=1)
    return [seq[:width] + [pad_id] * max(0, width - len(seq)) for seq in sequences]
