from __future__ import annotations

import pytest

from star_spatial.hd_matched_molecules import (
    MoleculeAssignment,
    compact_residual_components,
    exact_member_matches,
    match_membership_components,
)


def molecule(identifier: str, reads: tuple[str, ...]) -> MoleculeAssignment:
    return MoleculeAssignment(identifier, "g", "U", "s_002um_0_0", reads)


def test_exact_and_split_membership_components() -> None:
    star = [molecule("a", ("r1", "r2")), molecule("b", ("r3", "r4"))]
    sr = [
        molecule("x", ("r1", "r2")),
        molecule("y", ("r3",)),
        molecule("z", ("r4",)),
    ]
    assert [(left.molecule_id, right.molecule_id) for left, right in exact_member_matches(
        star, sr
    )] == [("a", "x")]
    classes = sorted(row.classification for row in match_membership_components(star, sr))
    assert classes == ["exact_member_set", "space_ranger_split"]

    matches = exact_member_matches(star, sr)
    residual = compact_residual_components(star, sr, matches)
    compact_classes = ["exact_member_set"] * len(matches) + [
        row.classification for row in residual
    ]
    assert sorted(compact_classes) == classes


def test_method_only_and_partial_components() -> None:
    star = [molecule("a", ("r1", "r2")), molecule("b", ("r3",))]
    sr = [molecule("x", ("r1",)), molecule("z", ("r4",))]
    classes = sorted(row.classification for row in match_membership_components(star, sr))
    assert classes == ["one_to_one_partial", "space_ranger_only", "star_only"]


def test_duplicate_molecule_identifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate molecule identifiers"):
        match_membership_components(
            [molecule("a", ("r1",)), molecule("a", ("r2",))], []
        )
