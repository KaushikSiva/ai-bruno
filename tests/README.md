# Tests

```bash
python3 -m unittest discover -s tests -v
```

`test_vla.py` covers the bounded control layer: kinematics, the action
contract, calibration fitting, arm state, the safety controller, and both
drivers. The MuJoCo cases skip themselves when the simulation extras are not
installed, so the suite still runs on Bruno.
