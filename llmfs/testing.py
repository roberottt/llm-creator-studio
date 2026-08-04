"""Helpers shared by the module tests.

The usual pattern for validating an `nn.Module` you wrote yourself:

    ref = ReferenceRMSNorm(320)
    mine = MyRMSNorm(320)
    copy_parameters(ref, mine)          # exactly the same weights
    assert_close(mine(x), ref(x))       # then the output must match

Copying the weights is essential: two modules with the same architecture but different
initializations produce different outputs and the test would say nothing. What is being
compared is the FUNCTION the module computes, not its initial values.
"""

from __future__ import annotations

from typing import Any

import torch


def load_exercises(test_file: str) -> Any:
    """Load the `exercises.py` of the module this test belongs to.

    Usage, always the same, at the top of every `test_NN.py`:

        from llmfs.testing import load_exercises
        ex = load_exercises(__file__)

    Unlike the bridge, there is NO safety net here: the tests validate YOUR code, so if an
    exercise is not done the test has to go red.

    Exception: with `LLMFS_TEST_REFERENCE=1` it returns `llmfs.reference` instead of your
    exercises. That makes the whole suite validate the course itself: if a test fails in
    that mode, either the test is badly written or the reference is wrong, and either way
    it is a bug in the repo and not yours. That is what `make test-reference` runs.
    """
    import os

    if os.environ.get("LLMFS_TEST_REFERENCE", "").strip().lower() in {"1", "true", "yes"}:
        import llmfs.reference

        return llmfs.reference

    from llmfs.bridge import exercises

    module = exercises(test_file)
    if module is None:
        raise AssertionError(
            "Could not import this module's exercises.py. "
            "Look at the traceback above: it is almost always a syntax error."
        )
    return module


def copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy the weights from `source` into `target`.

    Fails with a useful message if the names or shapes do not match, which is the usual way
    of discovering that your module has the architecture wired up wrong.
    """
    src_state = source.state_dict()
    dst_state = target.state_dict()

    only_ref = sorted(set(src_state) - set(dst_state))
    only_yours = sorted(set(dst_state) - set(src_state))
    if only_ref or only_yours:
        raise AssertionError(
            "The parameters do not match the reference's.\n"
            f"  missing in yours : {only_ref}\n"
            f"  extra in yours   : {only_yours}\n"
            "Check the submodule names in the exercise docstring."
        )

    for name, tensor in src_state.items():
        if tensor.shape != dst_state[name].shape:
            raise AssertionError(
                f"`{name}` has shape {tuple(dst_state[name].shape)} and should have "
                f"{tuple(tensor.shape)}."
            )
    target.load_state_dict(src_state)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    what: str = "the result",
) -> None:
    """`torch.allclose` with an error message you can actually debug with.

    Reports the shape, the maximum error and where it is, instead of the bare
    `assert False` that tells you nothing.
    """
    actual = actual if isinstance(actual, torch.Tensor) else torch.as_tensor(actual)
    expected = expected if isinstance(expected, torch.Tensor) else torch.as_tensor(expected)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what} has shape {tuple(actual.shape)} and {tuple(expected.shape)} was expected."
        )

    a32, e32 = actual.detach().float(), expected.detach().float()
    if torch.allclose(a32, e32, rtol=rtol, atol=atol):
        return

    diff = (a32 - e32).abs()
    idx = int(diff.argmax())
    pos = torch.unravel_index(torch.tensor(idx), diff.shape)
    pos_txt = tuple(int(p) for p in pos)
    raise AssertionError(
        f"{what} does not match the reference.\n"
        f"  shape           : {tuple(actual.shape)}\n"
        f"  max error       : {diff.max().item():.3e} at position {pos_txt}\n"
        f"  yours / expected: {a32.flatten()[idx].item():.6f} / {e32.flatten()[idx].item():.6f}\n"
        f"  mean error      : {diff.mean().item():.3e}"
    )


def assert_scalar_close(
    actual: Any, expected: Any, rtol: float = 1e-5, atol: float = 1e-6, what: str = "the value"
) -> None:
    """Like `assert_close` but for python scalars.

    It calls `detach()` on whatever arrives: the tests compare losses that usually come with
    an autograd graph attached, and `float()` on those tensors raises a UserWarning.
    """
    a = float(actual.detach()) if isinstance(actual, torch.Tensor) else float(actual)
    e = float(expected.detach()) if isinstance(expected, torch.Tensor) else float(expected)
    if abs(a - e) <= atol + rtol * abs(e):
        return
    raise AssertionError(f"{what}: {a!r}, expected {e!r} (error {abs(a - e):.3e})")


def seeded_generator(seed: int = 0, device: torch.device | str = "cpu") -> torch.Generator:
    """A reproducible generator. The tests compare against the reference, so it is needed."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    return gen
