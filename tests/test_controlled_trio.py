from scripts.train_foundation_generation_v1 import DATA_ORDER_SEED, MODELS, STAGE_ORDER


def test_trio_is_controlled_except_for_initialization_seed() -> None:
    assert tuple(name for name, _ in MODELS) == ("STATIC-T", "STATIC-X", "STATIC-K")
    seeds = [seed for _, seed in MODELS]
    assert len(set(seeds)) == 3
    assert DATA_ORDER_SEED == 20_261_017
    assert STAGE_ORDER == (
        "pretrain",
        "reasoning",
        "software",
        "instruction",
        "business",
        "finance",
        "relationship",
        "browser",
        "executive",
        "ethics",
        "judgment",
        "preference",
    )
