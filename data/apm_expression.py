"""Turn RNA expression into continuous APM and stimulus weights.

`ExcisionHead` accepts `apm_weights` and `stimulus_weights` -- mixtures over
`APM_PERTURBATIONS` and `PROCESSING_STIMULI` -- but nothing produced them. The
categorical path is fed from curated labels, which exist for a few hundred
samples and are absent for the rest. Expression is available far more widely,
and it is the quantity the labels are a proxy for: "peptide_supply perturbed"
is a coarse statement about TAP and immunoproteasome transcript levels.

This module is the missing caller. It is deliberately **deterministic and
parameter-free**: no learned projection stands between the transcript and the
weight, so a weight can be traced back to the genes that produced it and
checked. A learned encoder would be more expressive and much harder to falsify.

Direction matters and is not uniform across axes. The APM groups describe
**loss** of a processing component, so their weight rises as expression falls.
A stimulus describes **induction**, so its weight rises as the response
signature climbs. Treating both with one sign would read a TAP knockout as an
interferon response.

Calibrated against DepMap 24Q4, 1,673 lines, all 41 genes present. With the
shipped defaults 90% of lines put >0.8 on `none`, and an in-silico knockout of
`peptide_supply`, `n_term_trimming`, `loading_complex` or `mhc_null` drives its
own axis to 0.73-1.00 while every other axis moves by exactly 0.00.

Validated against genotype, which the layer never sees. Cross-referencing
DepMap's damaging-mutation matrix over 1,597 lines with both data types:

    axis                n dmg   mean w dmg   mean w wt      p (MWU)
    mhc_null               28        0.321       0.019      1.0e-22
    n_term_trimming        15        0.044       0.016      9.5e-03
    peptide_supply         14        0.022       0.014      0.17
    loading_complex        12        0.001       0.005      0.14
    class_ii_loading       21        0.014       0.008      0.53

B2M is the decisive case and it is overwhelming: 36% of B2M-damaged lines
exceed a weight of 0.3 on `mhc_null` against 2% of wild-type lines, a
seventeen-fold enrichment, from expression alone.

The pattern across axes is exactly what the mechanism predicts rather than
what one would hope. A damaging mutation is only visible here when it lowers
the transcript -- nonsense and frameshift alleles trigger nonsense-mediated
decay, missense alleles do not. So the layer is **specific but not sensitive**:
it rarely calls a perturbation that is not there (2% of wild-type lines), and
it misses the majority of damaging mutations because most leave mRNA intact.

For this use specificity is the property that matters, since a false
perturbation call corrupts the state vector for a cell that is actually
normal. But a low weight is never evidence a component is intact, and the
remaining axes are underpowered at n = 12-21 damaged lines regardless.

**Known limitation: `class_ii_loading` is weakly detected**, reaching only ~0.16
on a full in-silico knockout even in a B-cell line expressing all six genes.
Class-II loading genes vary far more widely among the cells that use them than
the class-I machinery does, so an abolition is a smaller number of SD. The axis
is reported, but a low weight on it is not evidence the pathway is intact.
Class-I work is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .vocab import APM_PERTURBATIONS, PROCESSING_STIMULI

#: Gene -> APM mechanism group. Mirrors `vocab.APM_GENE_TO_GROUP`, which maps
#: hitlist's per-gene perturbation flags onto the same groups. Keeping the two
#: consistent is what lets a curated label and an expression-derived weight
#: describe the same axis.
APM_GROUP_GENES: Dict[str, tuple[str, ...]] = {
    "peptide_supply": ("TAP1", "TAP2", "PSMB5", "PSMB8", "PSMB9", "PSMB10"),
    "n_term_trimming": ("ERAP1", "ERAP2"),
    "loading_complex": ("TAPBP", "TAPBPL", "CALR", "CANX", "PDIA3"),
    "mhc_null": ("B2M",),
    "class_ii_loading": ("CIITA", "CD74", "HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB"),
}

#: Interferon-gamma response genes. Induction markers, not machinery: the
#: machinery genes are already the APM axis, and scoring them twice would let a
#: single transcript drive both a "peptide supply is intact" and an "interferon
#: is present" conclusion from the same evidence.
STIMULUS_GROUP_GENES: Dict[str, tuple[str, ...]] = {
    "ifn_gamma": ("GBP1", "GBP2", "CXCL9", "CXCL10", "CXCL11", "IDO1", "STAT1", "IRF1", "SOCS1"),
    "ifn_type1": ("ISG15", "IFI6", "MX1", "OAS1", "IFIT1", "IFIT3", "RSAD2"),
    "tnf_alpha": ("TNFAIP3", "NFKBIA", "CCL2", "ICAM1", "BIRC3"),
}

#: Groups whose weight rises when expression FALLS.
LOSS_GROUPS = frozenset(APM_GROUP_GENES)


@dataclass(frozen=True)
class ExpressionReference:
    """Per-gene cohort mean and standard deviation, on a log scale.

    Built once from a reference cohort (for example every DepMap line) so a
    sample's deviation is expressed in cohort standard deviations rather than
    in absolute TPM, which is not comparable across studies or platforms.
    """

    mean: Mapping[str, float]
    sd: Mapping[str, float]
    #: Genes whose cohort SD is below this are treated as unmeasurable rather
    #: than as infinitely informative; dividing by a near-zero SD turns
    #: quantisation noise into a full-severity call.
    min_sd: float = 0.05

    def z(self, gene: str, value: float) -> Optional[float]:
        if gene not in self.mean or gene not in self.sd:
            return None
        sd = self.sd[gene]
        if sd is None or sd < self.min_sd:
            return None
        return (value - self.mean[gene]) / sd

    @classmethod
    def from_matrix(
        cls,
        matrix: Mapping[str, Sequence[float]],
        min_sd: float = 0.05,
        expressor_floor: Optional[float] = 1.0,
        min_expressors: int = 30,
    ) -> "ExpressionReference":
        """`matrix` maps gene -> values across the reference cohort.

        `expressor_floor` conditions the reference on cells that actually use
        the gene. Without it a gene most of the cohort does not express is
        undetectable when it IS lost: across 1,673 DepMap lines the class-II
        loading genes sit near the floor in most cells, so the pooled mean is
        already low and the SD is wide. Abolishing all six in a B-cell line --
        a total knockout of the group -- moved the mean z only to -0.95, under
        any usable threshold. The knockout was invisible because the cohort
        baseline was "off".

        Conditioning on expressors makes the SD describe variation among cells
        that run the pathway, which is the variation a perturbation perturbs.
        Genes with fewer than `min_expressors` above the floor are dropped
        rather than described by a handful of points. Set `expressor_floor=None`
        to pool everything.
        """
        mean, sd = {}, {}
        for gene, values in matrix.items():
            vals = [v for v in values if v is not None]
            if expressor_floor is not None:
                expressed = [v for v in vals if v > expressor_floor]
                if len(expressed) >= min_expressors:
                    vals = expressed
                else:
                    continue
            if len(vals) < 2:
                continue
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            mean[gene] = m
            sd[gene] = var**0.5
        return cls(mean=mean, sd=sd, min_sd=min_sd)


@dataclass
class WeightBuilder:
    """Expression -> `(apm_weights, stimulus_weights)`.

    `scale` is the number of cohort standard deviations that counts as a fully
    engaged axis. 2.0 means a group two SD below the cohort reads as completely
    perturbed. It is a stated assumption, not a fitted value, and the honest
    way to use it is to check that conclusions survive a sweep.
    """

    reference: ExpressionReference
    #: Deviation, in cohort SD, below which a group is called unperturbed --
    #: severity is exactly zero, not merely small.
    #:
    #: Without it the layer calls a perturbation on every line. Across 1,673
    #: DepMap lines a plain linear ramp put mean `none` at 0.49 and gave HeLa
    #: an n_term_trimming weight of 0.48, which is not a claim anyone would
    #: defend. Ordinary expression variation among cancer lines is one to two
    #: SD wide, so a ramp starting at zero reads that variation as loss of
    #: function.
    #:
    #: A deadband also matches the biology better than a pure continuum. The
    #: labelled states this axis mixes over are knockouts and inhibitor arms --
    #: qualitative events. The continuous axis exists to express *degree* once a
    #: component is genuinely affected, not to assert that every cell is
    #: slightly TAP-deficient.
    deadband: float = 1.5
    #: Cohort SD above the deadband at which the axis is fully engaged.
    scale: float = 2.0
    #: Fraction of a group's genes that must be measurable before the group is
    #: scored. Below it the group contributes nothing and `unknown` absorbs the
    #: mass, which is a different claim from "not perturbed".
    min_coverage: float = 0.5
    apm_groups: Dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(APM_GROUP_GENES))
    stimulus_groups: Dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(STIMULUS_GROUP_GENES)
    )

    def _severity(self, genes: Iterable[str], expr: Mapping[str, float], loss: bool):
        zs = []
        genes = tuple(genes)
        for g in genes:
            if g not in expr:
                continue
            z = self.reference.z(g, expr[g])
            if z is not None:
                zs.append(z)
        if not genes or len(zs) / len(genes) < self.min_coverage:
            return None
        mean_z = sum(zs) / len(zs)
        signed = -mean_z if loss else mean_z
        # Deadband first, then ramp over `scale` SD beyond it.
        excess = signed - self.deadband
        if excess <= 0.0:
            return 0.0
        return max(0.0, min(1.0, excess / self.scale))

    def _normalize(self, severities: Dict[str, float], vocab: Sequence[str]):
        """Convex combination over `vocab`, with the residue on `none`.

        Severities are capped at 1 each but several axes can be engaged at once,
        so the total may exceed 1. Rescaling in that case, rather than clipping,
        keeps the ratio between axes -- which is the informative part -- and
        only gives up the absolute magnitude, which the profile tables' own
        scale absorbs anyway.
        """
        weights = {k: 0.0 for k in vocab}
        total = sum(severities.values())
        if total > 1.0:
            severities = {k: v / total for k, v in severities.items()}
            total = 1.0
        for k, v in severities.items():
            if k in weights:
                weights[k] = v
        weights["none"] = max(0.0, 1.0 - total)
        return [weights[k] for k in vocab]

    def apm_weights(self, expr: Mapping[str, float]) -> list[float]:
        sev, unmeasured = {}, 0
        for group, genes in self.apm_groups.items():
            s = self._severity(genes, expr, loss=group in LOSS_GROUPS)
            if s is None:
                unmeasured += 1
            else:
                sev[group] = s
        w = self._normalize(sev, APM_PERTURBATIONS)
        # Every group unmeasurable is not evidence of an intact cell. Put the
        # mass on `unknown`, which the vocabulary added for exactly this case.
        if unmeasured == len(self.apm_groups):
            w = [0.0] * len(APM_PERTURBATIONS)
            w[APM_PERTURBATIONS.index("unknown")] = 1.0
        return w

    def stimulus_weights(self, expr: Mapping[str, float]) -> list[float]:
        sev = {}
        for group, genes in self.stimulus_groups.items():
            s = self._severity(genes, expr, loss=False)
            if s is not None:
                sev[group] = s
        return self._normalize(sev, PROCESSING_STIMULI)

    def weights(self, expr: Mapping[str, float]):
        return self.apm_weights(expr), self.stimulus_weights(expr)
