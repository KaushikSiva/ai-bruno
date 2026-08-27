"""Bounded, calibrated control of Bruno's chassis and arm, in sim or for real.

One action contract, one safety controller, and three interchangeable drivers
(mock, MuJoCo, MasterPi hardware), all sharing a single calibration profile so
a command means the same physical motion whichever backend executes it.
"""

from .calibration import CalibrationProfile, load_profile, save_profile
from .contracts import ALLOWED_ACTIONS, Action, ContractError
from .controller import RobotController, RobotNotArmed

__all__ = [
    "ALLOWED_ACTIONS",
    "Action",
    "CalibrationProfile",
    "ContractError",
    "RobotController",
    "RobotNotArmed",
    "load_profile",
    "save_profile",
]
