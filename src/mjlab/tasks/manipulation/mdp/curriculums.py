from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from mjlab.tasks.manipulation.mdp.commands import PickPlaceCommandCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_REQUIRED_STAGE_KEYS = frozenset({"step", "radius", "tol"})


class pick_place_curriculum:
  """Widen the pick-to-place distance and tighten the placement tolerance.

  Two axes advance together on a step schedule:

  * ``radius`` sets :attr:`PickPlaceCommandCfg.goal_radius_range`, the annulus
    the floor goal is drawn from around the object.
  * ``tol`` sets :attr:`PickPlaceCommandCfg.place_tol`, the success radius, and
    simultaneously the ``std`` of the sharp placement reward.

  Tying ``std`` to ``tol`` is load-bearing rather than tidy. The Gaussian width
  of the placement reward and the success radius are independent parameters, so
  shrinking the radius on its own adds no extra gradient pulling the cube
  inward -- it only stops the bonus firing, and from the policy's side that is
  indistinguishable from having gotten worse.

  Mutation targets differ by manager. ``CommandManager`` does not deepcopy its
  config (``command_manager.py:262-273``), so the command config object reached
  through the live term is the one the sampler reads. ``RewardManager`` *does*
  deepcopy, so its term has to be fetched with ``get_term_cfg`` -- assigning to
  ``env.cfg.rewards[...]`` would silently do nothing.

  Timing. ``curriculum_manager.compute`` runs only from ``_reset_idx``
  (``manager_based_rl_env.py:554``), so a stage takes effect at the next reset
  after its step threshold rather than exactly on it, and ``step`` counts
  environment steps: one iteration is ``num_steps_per_env`` (24) of them.

  Example::

    CurriculumTermCfg(
      func=mdp.pick_place_curriculum,
      params={
        "command_name": "pick_place",
        "place_reward_name": "place",
        "stages": [
          {"step": 0, "radius": (0.15, 0.20), "tol": 0.10},
          {"step": 1500 * 24, "radius": (0.15, 0.32), "tol": 0.06},
          {"step": 3000 * 24, "radius": (0.15, 0.45), "tol": 0.04},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    place_reward_name: str = cfg.params["place_reward_name"]
    stages: Sequence[dict[str, Any]] = cfg.params["stages"]

    command_cfg = getattr(env.command_manager.get_term(command_name), "cfg", None)
    if not isinstance(command_cfg, PickPlaceCommandCfg):
      raise TypeError(
        f"Command '{command_name}' must be configured by a PickPlaceCommandCfg, "
        f"got {type(command_cfg)}"
      )
    self._command_cfg = command_cfg
    self._place_cfg = env.reward_manager.get_term_cfg(place_reward_name)
    if "std" not in self._place_cfg.params:
      raise KeyError(
        f"Reward term '{place_reward_name}' has no 'std' param for the "
        "placement curriculum to track."
      )
    self._stages = sorted(stages, key=lambda s: s["step"])
    self._validate()

  def _validate(self) -> None:
    for stage in self._stages:
      missing = _REQUIRED_STAGE_KEYS - stage.keys()
      unknown = stage.keys() - _REQUIRED_STAGE_KEYS
      if missing or unknown:
        raise KeyError(
          f"pick_place_curriculum stage at step {stage.get('step')} is "
          f"malformed: missing {sorted(missing)}, unknown {sorted(unknown)}."
        )
      low, high = stage["radius"]
      if low > high:
        raise ValueError(
          f"pick_place_curriculum stage at step {stage['step']} has an "
          f"inverted radius range ({low}, {high})."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    place_reward_name: str,
    stages: Sequence[dict[str, Any]],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, place_reward_name, stages
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        self._command_cfg.goal_radius_range = tuple(stage["radius"])
        self._command_cfg.place_tol = stage["tol"]
        self._place_cfg.params["std"] = stage["tol"]
    return {
      "goal_radius_min": torch.tensor(self._command_cfg.goal_radius_range[0]),
      "goal_radius_max": torch.tensor(self._command_cfg.goal_radius_range[1]),
      "place_tol": torch.tensor(self._command_cfg.place_tol),
    }
