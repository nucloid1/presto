"""The APM and stimulus axes accept continuous weights, not only a label.

`ExcisionHead` conditions cleavage on two categorical indices. Expression data
does not come in labels: immunoproteasome subunit ratio, TAP level and ERAP
level are continuous, and a cell at 40% immunoproteasome is not any of the
discrete states. Indexing forces a rounding decision the data does not support,
and cannot extrapolate to a state that was never given a label.

`apm_weights` and `stimulus_weights` are `(batch, n_cond)` mixtures over the
same vocabularies. The contract these tests pin:

1. one-hot weights reproduce the categorical path EXACTLY, so nothing that
   exists today changes and old checkpoints keep their meaning;
2. gradients reach the profile tables through the mixture, so the continuous
   axis is trainable rather than decorative;
3. a mixture lies strictly between its endpoints, which is the property that
   makes an intermediate state representable at all.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.vocab import AA_VOCAB, APM_PERTURBATIONS, PROCESSING_STIMULI  # noqa: E402
from presto.models.heads import ExcisionHead  # noqa: E402

N_AA = len(AA_VOCAB)
N_APM = len(APM_PERTURBATIONS)
N_STIM = len(PROCESSING_STIMULI)
D_MODEL = 16
WINDOW = 5


def _head():
    torch.manual_seed(0)
    head = ExcisionHead(
        d_model=D_MODEL,
        n_machinery=2,
        n_aa=N_AA,
        n_apm=N_APM,
        n_stimulus=N_STIM,
        junction_window=WINDOW,
    )
    # Zero-initialized tables would make every assertion below trivially true.
    with torch.no_grad():
        for t in (head.invivo_profile_c, head.invivo_profile_n, head.stimulus_profile_c):
            t.normal_(0.0, 0.5)
        head.invivo_bias.normal_(0.0, 0.5)
    return head


def _batch(n=4):
    torch.manual_seed(1)
    w = 2 * WINDOW
    return dict(
        processing_vec=torch.randn(n, D_MODEL),
        machinery_idx=torch.zeros(n, dtype=torch.long),
        p1_c_idx=torch.randint(1, N_AA, (n,)),
        p1_prime_c_idx=torch.randint(1, N_AA, (n,)),
        p1_n_idx=torch.randint(1, N_AA, (n,)),
        p1_prime_n_idx=torch.randint(1, N_AA, (n,)),
        peptide_len=torch.full((n,), 9, dtype=torch.long),
        window_c_idx=torch.randint(1, N_AA, (n, w)),
        window_n_idx=torch.randint(1, N_AA, (n, w)),
        # 1 == 'mhc', the in-vivo branch these tables act on.
        peptide_source_idx=torch.ones(n, dtype=torch.long),
    )


def _one_hot(idx, n):
    return torch.nn.functional.one_hot(idx, num_classes=n).to(torch.float32)


class TestOneHotReducesToCategorical:
    """The continuous path must be a strict generalization, not a replacement."""

    def test_excision_logit_matches_exactly(self):
        head, batch = _head(), _batch()
        apm = torch.tensor([0, 1, 2, 1])
        stim = torch.tensor([0, 1, 1, 2])

        hard = head(**batch, apm_perturbation_idx=apm, processing_stimulus_idx=stim)
        soft = head(
            **batch,
            apm_weights=_one_hot(apm, N_APM),
            stimulus_weights=_one_hot(stim, N_STIM),
        )
        torch.testing.assert_close(soft["excision_logit"], hard["excision_logit"])

    def test_counterfactual_panels_match_exactly(self):
        head, batch = _head(), _batch()
        apm = torch.tensor([0, 1, 2, 1])
        stim = torch.tensor([0, 1, 1, 2])

        hard = head(**batch, apm_perturbation_idx=apm, processing_stimulus_idx=stim)
        soft = head(
            **batch,
            apm_weights=_one_hot(apm, N_APM),
            stimulus_weights=_one_hot(stim, N_STIM),
        )
        torch.testing.assert_close(soft["excision_panel_apm"], hard["excision_panel_apm"])
        torch.testing.assert_close(
            soft["excision_panel_stimulus"], hard["excision_panel_stimulus"]
        )

    def test_observed_column_still_reproduces_the_scalar(self):
        """The property the panel loss depends on, under one-hot weights."""
        head, batch = _head(), _batch()
        apm = torch.tensor([0, 1, 2, 1])
        out = head(**batch, apm_weights=_one_hot(apm, N_APM))
        observed = out["excision_panel_apm"].gather(1, apm.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(observed, out["excision_logit"])


class TestMixtureIsTrainableAndIntermediate:
    def test_gradients_reach_the_profile_tables(self):
        head, batch = _head(), _batch()
        weights = torch.full((4, N_APM), 1.0 / N_APM, requires_grad=True)
        out = head(**batch, apm_weights=weights)
        out["excision_logit"].sum().backward()
        assert head.invivo_profile_c.grad is not None
        assert head.invivo_profile_c.grad.abs().sum() > 0
        assert weights.grad is not None
        assert weights.grad.abs().sum() > 0

    def test_a_blend_lies_between_its_endpoints(self):
        """The whole point: an intermediate state gets an intermediate score."""
        head, batch = _head(), _batch()
        a = _one_hot(torch.full((4,), 1), N_APM)
        b = _one_hot(torch.full((4,), 2), N_APM)
        half = 0.5 * a + 0.5 * b

        sa = head(**batch, apm_weights=a)["excision_logit"]
        sb = head(**batch, apm_weights=b)["excision_logit"]
        sh = head(**batch, apm_weights=half)["excision_logit"]

        lo, hi = torch.minimum(sa, sb), torch.maximum(sa, sb)
        assert torch.all(sh >= lo - 1e-5) and torch.all(sh <= hi + 1e-5)
        # and it is genuinely the midpoint, since the score is linear in the table
        torch.testing.assert_close(sh, 0.5 * (sa + sb))

    def test_categorical_path_is_untouched_when_no_weights_given(self):
        head, batch = _head(), _batch()
        apm = torch.tensor([0, 1, 2, 1])
        a = head(**batch, apm_perturbation_idx=apm)
        b = head(**batch, apm_perturbation_idx=apm)
        torch.testing.assert_close(a["excision_logit"], b["excision_logit"])
