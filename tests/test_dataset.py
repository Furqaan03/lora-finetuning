from src.data.dataset import Example, assert_no_leakage, clean_examples, split_dataset


def _examples(n):
    return [Example(instruction=f"task {i}", input=f"input {i}", output=f"out {i}") for i in range(n)]


def test_clean_removes_empties():
    exs = [Example("do x", "", "result"), Example("", "in", "out"), Example("do y", "", "")]
    cleaned = clean_examples(exs)
    assert len(cleaned) == 1
    assert cleaned[0].instruction == "do x"


def test_clean_removes_duplicates():
    exs = [Example("same", "in", "out1"), Example("same", "in", "out2")]  # same instruction+input
    assert len(clean_examples(exs)) == 1


def test_prompt_format_with_and_without_input():
    with_input = Example("classify", "some text", "label").to_prompt()
    assert "### Input:" in with_input
    without_input = Example("say hi", "", "hi").to_prompt()
    assert "### Input:" not in without_input


def test_split_ratios():
    splits = split_dataset(_examples(100))
    assert len(splits.train) == 80
    assert len(splits.val) == 10
    assert len(splits.test) == 10


def test_split_is_deterministic():
    a = split_dataset(_examples(50))
    b = split_dataset(_examples(50))
    assert [e.instruction for e in a.test] == [e.instruction for e in b.test]


def test_no_leakage_across_splits():
    splits = split_dataset(_examples(100))
    assert_no_leakage(splits)  # raises if leakage
