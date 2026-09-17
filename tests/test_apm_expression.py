"""Expression -> continuous APM/stimulus weights.

These use a synthetic reference cohort so the tests are hermetic. The layer was
developed and calibrated against DepMap 24Q4 (1,673 lines, 41 genes); the
numbers quoted in `data/apm_expression.py` come from that run.

Two calibration findings are pinned here because both were real bugs caught by
validating against that cohort rather than by reasoning:

1. Without a deadband the layer calls a perturbation on nearly every cell.
2. Without an expressor-conditional reference a knockout is invisible whenever
   most of the cohort does not express the gene.
"""

import pytest

from presto.data.apm_expression import (  # noqa: E402
    APM_GROUP_GENES,
    ExpressionReference,
    WeightBuilder,
)
from presto.data.vocab import APM_PERTURBATIONS, PROCESSING_STIMULI  # noqa: E402


def _reference(genes, mean=6.0, sd=1.0, n=200):
    """A cohort where every gene sits at `mean` with spread `sd`."""
    import random

    random.seed(0)
    matrix = {g: [random.gauss(mean, sd) for _ in range(n)] for g in genes}
    return ExpressionReference.from_matrix(matrix)


ALL_APM_GENES = tuple(g for gs in APM_GROUP_GENES.values() for g in gs)


def _typical(genes, value=6.0):
    return {g: value for g in genes}


class TestUnperturbedCellsReadAsUnperturbed:
    def test_a_cohort_typical_cell_puts_all_mass_on_none(self):
        ref = _reference(ALL_APM_GENES)
        wb = WeightBuilder(reference=ref)
        w = wb.apm_weights(_typical(ALL_APM_GENES))
        assert w[APM_PERTURBATIONS.index("none")] == pytest.approx(1.0)

    def test_ordinary_variation_does_not_trip_the_deadband(self):
        """One SD low is normal variation, not a perturbation."""
        ref = _reference(ALL_APM_GENES)
        wb = WeightBuilder(reference=ref, deadband=1.5)
        expr = _typical(ALL_APM_GENES)
        for g in APM_GROUP_GENES["peptide_supply"]:
            expr[g] = 5.0  # exactly 1 SD below
        assert wb.apm_weights(expr)[APM_PERTURBATIONS.index("none")] == pytest.approx(1.0)

    def test_without_a_deadband_the_same_cell_is_called_perturbed(self):
        """Pins why the deadband exists."""
        ref = _reference(ALL_APM_GENES)
        expr = _typical(ALL_APM_GENES)
        for g in APM_GROUP_GENES["peptide_supply"]:
            expr[g] = 5.0
        w = WeightBuilder(reference=ref, deadband=0.0).apm_weights(expr)
        assert w[APM_PERTURBATIONS.index("peptide_supply")] > 0.0


class TestLossIsDetectedOnTheRightAxis:
    @pytest.mark.parametrize("group", ["peptide_supply", "n_term_trimming", "loading_complex", "mhc_null"])
    def test_knocking_down_one_group_moves_only_that_group(self, group):
        ref = _reference(ALL_APM_GENES)
        wb = WeightBuilder(reference=ref)
        expr = _typical(ALL_APM_GENES)
        for g in APM_GROUP_GENES[group]:
            expr[g] = 0.0
        w = dict(zip(APM_PERTURBATIONS, wb.apm_weights(expr)))
        assert w[group] > 0.5
        for other in APM_PERTURBATIONS:
            if other in (group, "none"):
                continue
            assert w[other] == pytest.approx(0.0), f"{group} knockdown leaked into {other}"

    def test_weights_form_a_convex_combination(self):
        ref = _reference(ALL_APM_GENES)
        wb = WeightBuilder(reference=ref)
        expr = _typical(ALL_APM_GENES)
        for g in APM_GROUP_GENES["peptide_supply"] + APM_GROUP_GENES["mhc_null"]:
            expr[g] = 0.0
        w = wb.apm_weights(expr)
        assert all(x >= 0.0 for x in w)
        assert sum(w) == pytest.approx(1.0)


class TestDirectionIsPerAxis:
    def test_induction_raises_a_stimulus_rather_than_lowering_it(self):
        """A loss sign on the stimulus axis would read IFN-gamma as a knockout."""
        from presto.data.apm_expression import STIMULUS_GROUP_GENES

        genes = tuple(g for gs in STIMULUS_GROUP_GENES.values() for g in gs)
        ref = _reference(genes)
        wb = WeightBuilder(reference=ref)
        expr = _typical(genes)
        for g in STIMULUS_GROUP_GENES["ifn_gamma"]:
            expr[g] = 12.0  # strongly induced
        w = dict(zip(PROCESSING_STIMULI, wb.stimulus_weights(expr)))
        assert w["ifn_gamma"] > 0.5
        assert w["none"] < 0.5


class TestUnmeasurableIsNotUnperturbed:
    def test_no_measurable_genes_goes_to_unknown_not_none(self):
        ref = _reference(ALL_APM_GENES)
        w = dict(zip(APM_PERTURBATIONS, WeightBuilder(reference=ref).apm_weights({})))
        assert w["unknown"] == pytest.approx(1.0)
        assert w["none"] == pytest.approx(0.0)


class TestExpressorConditionalReference:
    def test_knockout_is_visible_when_most_of_the_cohort_is_off(self):
        """The class-II failure mode, in miniature.

        A gene 90% of the cohort does not express: pooling everything makes the
        mean low and the SD wide, so abolishing it in an expressing cell barely
        registers. Conditioning on expressors recovers it.
        """
        import random

        random.seed(1)
        vals = [0.0] * 500 + [random.gauss(8.0, 0.5) for _ in range(100)]
        pooled = ExpressionReference.from_matrix({"G": vals}, expressor_floor=None)
        conditioned = ExpressionReference.from_matrix({"G": vals}, expressor_floor=1.0)
        assert abs(pooled.z("G", 0.0)) < abs(conditioned.z("G", 0.0))
        assert abs(conditioned.z("G", 0.0)) > 3.0
