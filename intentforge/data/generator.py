"""Synthetic intent-classification dataset generator.

Deterministic given ``random_seed`` so benchmarks are reproducible with zero
network access.

Design notes
------------
Each intent has a template list and a class-specific phrase pool; every sample
combines one template + one pool phrase + optional filler/noise. This yields
*many distinct surface forms* per class instead of N copies of one sentence,
which matters because:

* it removes train/test lexical duplication (no leakage -> honest accuracy);
* it creates a real difficulty gradient (short/clean sentences are easy,
  noisy blends are hard), so the C3R cascade has something to route.

A small injected set of *ambiguous* cross-intent sentences guarantees that some
samples genuinely confuse the fast model and must be escalated.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "{p}",
        "{p} how are you doing today",
        "{p} i just signed up",
        "{p} can you help me with something",
    ],
    "query_balance": [
        "{p}",
        "{p} right now please",
        "{p} how much money is left",
        "can you {p} for my checking account",
    ],
    "transfer_money": [
        "{p}",
        "{p} immediately",
        "{p} to another bank",
        "i need to {p} today",
    ],
    "complaint": [
        "{p}",
        "{p} this is unacceptable",
        "honestly {p} and nobody helps",
        "{p} worst experience ever",
    ],
    "cancel_subscription": [
        "{p}",
        "{p} effective immediately",
        "please {p} and refund me",
        "i want to {p} this month",
    ],
    "technical_support": [
        "{p}",
        "{p} since the update",
        "how do i fix this {p}",
        "{p} on the mobile app",
    ],
    "human_agent": [
        "{p}",
        "{p} please",
        "let me {p} instead of this bot",
        "{p} about my issue",
    ],
    "payment_issue": [
        "{p}",
        "{p} and i was still charged",
        "why did this happen {p}",
        "{p} can you check it",
    ],
    "account_access": [
        "{p}",
        "{p} again",
        "help i {p}",
        "{p} and i am worried",
    ],
    "product_info": [
        "{p}",
        "{p} in detail",
        "can you {p}",
        "questions about {p}",
    ],
}

_POOLS: Dict[str, List[str]] = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
        "hiya", "whats up", "greetings",
    ],
    "query_balance": [
        "check my balance", "show my account balance", "what is my current balance",
        "display my available funds", "tell me my balance", "account balance inquiry",
    ],
    "transfer_money": [
        "transfer money", "send two hundred dollars", "move funds to savings",
        "wire money abroad", "transfer cash to john", "pay my rent from checking",
    ],
    "complaint": [
        "this service is terrible", "your app keeps crashing", "i am very unhappy",
        "support ignored me for days", "the new update broke everything",
    ],
    "cancel_subscription": [
        "cancel my subscription", "stop auto renewal", "unsubscribe from premium",
        "end my monthly plan", "cancel the service",
    ],
    "technical_support": [
        "the login page shows an error", "my app will not sync", "reset my password",
        "the button is greyed out", "the website freezes on load",
    ],
    "human_agent": [
        "speak to a human agent", "talk to a real person", "connect me to live support",
        "get a representative on the line", "transfer me to an operator",
    ],
    "payment_issue": [
        "my payment was declined", "i was doubled billed", "the charge failed",
        "the refund never arrived", "money left my bank but order failed",
    ],
    "account_access": [
        "i am locked out of my account", "the two factor code never arrives",
        "i cannot log in with the right password", "someone accessed my account",
        "my session expired and i cannot get in",
    ],
    "product_info": [
        "the premium plan features", "the analytics dashboard", "annual billing discounts",
        "the starter versus pro tiers", "how pricing works",
    ],
}

# Ambiguous / cross-intent sentences -> genuine hard cases.
_AMBIGUOUS: List[str] = [
    "i need help with my account and there is also a payment problem",
    "can you check my balance and also cancel the plan today",
    "the app is slow and i cannot log in to transfer money",
    "i want to talk to someone about a wrong charge on my account",
    "my login failed and now i see a payment error",
    "please refund me and stop the subscription immediately",
    "why is my balance wrong after that transfer",
    "i am locked out and also got double billed somehow",
    "cancel my plan because the payments keep failing",
    "transfer money failed and i cannot access my balance",
]

_FILLERS: List[str] = [
    "please", "thanks", "as soon as possible", "today", "i am in a hurry",
    "thanks a lot", "cheers", "any help appreciated",
]

_NOISE: List[str] = [
    "it is raining outside", "my cat is on the keyboard", "this is unrelated but hi",
    "sorry for the bother", "hope you are well",
]


def generate_texts(cfg) -> Tuple[List[str], List[str]]:
    """Return ``(texts, labels)`` for roughly ``cfg.n_samples`` instances."""
    rng = random.Random(cfg.random_seed)
    labels = list(cfg.label_names)[: cfg.n_classes]
    if len(labels) < 2:
        raise ValueError("need at least 2 intent classes")

    # Split the requested budget: 80% clean per-class samples,
    # 20% ambiguous cross-intent samples, totalling exactly n_samples.
    total = int(cfg.n_samples)
    amb_count = int(0.20 * total)
    base_count = total - amb_count
    per_class = max(1, base_count // len(labels))

    texts: List[str] = []
    out_labels: List[str] = []

    for lab in labels:
        tmpls = _TEMPLATES.get(lab, ["{p}"])
        pool = _POOLS.get(lab, [lab])
        for _ in range(per_class):
            sentence = rng.choice(tmpls).replace("{p}", rng.choice(pool))
            if rng.random() < 0.35:
                sentence = f"{sentence} {rng.choice(_FILLERS)}"
            if rng.random() < 0.10:
                sentence = f"{rng.choice(_NOISE)} {sentence}"
            texts.append(sentence)
            out_labels.append(lab)

    # top up / truncate to exactly base_count (integer rounding)
    while len(texts) < base_count:
        lab = rng.choice(labels)
        sentence = rng.choice(_TEMPLATES.get(lab, ["{p}"])).replace(
            "{p}", rng.choice(_POOLS.get(lab, [lab]))
        )
        texts.append(sentence)
        out_labels.append(lab)
    texts = texts[:base_count]
    out_labels = out_labels[:base_count]

    # Inject cross-intent ambiguous samples (hard cases for the cascade).
    for _ in range(max(0, total - len(texts))):
        base = rng.choice(_AMBIGUOUS)
        if rng.random() < 0.3:
            base = f"{base} {rng.choice(_FILLERS)}"
        texts.append(base)
        out_labels.append(rng.choice(labels))

    paired = list(zip(texts, out_labels))
    rng.shuffle(paired)
    if paired:
        texts, out_labels = zip(*paired)
    return list(texts), list(out_labels)


def make_splits(texts: List[str], labels: List[str], cfg):
    """Stratified train/test split + train/val split for router calibration."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=cfg.test_size,
        random_state=cfg.random_seed, stratify=labels,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.val_size,
        random_state=cfg.random_seed, stratify=y_tr,
    )
    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_te, "y_test": y_te,
        "labels": sorted(set(labels)),
    }
