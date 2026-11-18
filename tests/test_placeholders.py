"""Placeholder filling tests."""

from grainofsalt.corpus import fill_template


def test_fill_asserted():
    out = fill_template(
        "{NAME} was born in {VALUE}.",
        name="Ada Test",
        value="Paris",
    )
    assert out == "Ada Test was born in Paris."


def test_fill_fiction_title():
    out = fill_template(
        "In the novel {TITLE}, {NAME} was born in {VALUE}.",
        name="Ada Test",
        value="Paris",
        title="Red Moons",
    )
    assert "Red Moons" in out and "Ada Test" in out and "Paris" in out


def test_unfilled_raises():
    try:
        fill_template("{NAME} met {VALUE} in {TITLE}.", name="A", value="B")
        assert False, "expected ValueError"
    except ValueError:
        pass
