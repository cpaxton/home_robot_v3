"""Critic and reward contact features use owned arrays, never binding object slices."""

from types import SimpleNamespace

import numpy as np

from emet.rl.manipulation.whole_body_observation import PRIVILEGED_FIELDS, privileged_state
from emet.rl.manipulation.whole_body_reward import current_contact_force


def test_critic_snapshots_active_contact_arrays_and_preserves_load_units():
    class Contacts:
        geom1 = np.array([0, 2, 4, -999])
        geom2 = np.array([1, 3, 0, -999])
        efc_address = np.array([0, 3, 6, 999])
        dim = np.full(4, 3)
        exclude = np.zeros(4, int)
        friction = np.ones((4, 5))

        def __len__(self):
            return 4

        def __getitem__(self, item):
            raise AssertionError("individual C-backed contact objects must not be accessed")

        def __iter__(self):
            raise AssertionError("individual C-backed contact objects must not be iterated")

    identities = {"table": 3, "left": 4, "right": 5, "wheel": 6}
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=identities[name]),
        geom_bodyid=np.array([2, 3, 6, 0, 4]),
        ngeom=5,
        opt=SimpleNamespace(cone=1),
        body_mass=np.full(8, 0.05),
        body_geomadr=np.zeros(8, int),
        geom_friction=np.ones((5, 3)),
    )
    data = SimpleNamespace(
        contact=Contacts(),
        ncon=3,
        nefc=9,
        efc_force=np.array([3, 0, 0, 2, 0, 0, 4, 1, -2]),
        xpos=np.zeros((8, 3)),
        xmat=np.tile(np.eye(3).reshape(1, 9), (8, 1)),
        xfrc_applied=np.zeros((8, 6)),
    )
    pose = np.r_[np.zeros(3), np.eye(3)[:, :2].ravel()]
    control = SimpleNamespace(
        ee=1,
        base=7,
        profile=SimpleNamespace(finger_bodies=("left", "right"), wheel_bodies=("wheel",)),
        retain_object=True,
        hand_goal=pose,
        ee_pose=lambda: pose,
    )
    env = SimpleNamespace(control=control, model=model, data=data, payload=2)
    features = privileged_state(env)

    def field(name):
        return features[slice(*PRIVILEGED_FIELDS[name])]

    np.testing.assert_array_equal(field("finger_contact_magnitudes"), [4, 3, 1, 0, 0, 0])
    np.testing.assert_array_equal(field("wheel_normal_loads_N"), [2, 0, 0, 0])
    np.testing.assert_array_equal(field("payload_support_normal_N"), [3])
    # Allocated inactive slots stay ignored, including their invalid geometry.
    data.ncon = 0
    empty = privileged_state(env)
    np.testing.assert_array_equal(empty[7:13], 0)
    np.testing.assert_array_equal(empty[19:24], 0)


def test_reward_contact_force_snapshots_active_arrays_and_never_constructs_objects():
    class Contacts:
        geom1 = np.array([0])
        geom2 = np.array([1])
        efc_address = np.array([0])
        dim = np.array([3])
        exclude = np.array([0])
        friction = np.ones((1, 5))

        def __len__(self):
            return 1

        def __getitem__(self, item):
            raise AssertionError("individual C-backed contact objects must not be accessed")

        def __iter__(self):
            raise AssertionError("individual C-backed contact objects must not be iterated")

    model = SimpleNamespace(
        geom_bodyid=np.array([2, 3]),
        ngeom=2,
        opt=SimpleNamespace(cone=1),  # elliptic: bound is the L1 norm of local force.
    )
    data = SimpleNamespace(
        contact=Contacts(),
        ncon=1,
        nefc=3,
        efc_force=np.array([2.0, 0.0, 4.0]),
    )
    monitor = SimpleNamespace(robot_geoms={0}, support_geoms=set(), floor={-1})
    env = SimpleNamespace(monitor=monitor, model=model, data=data, payload=3)
    assert current_contact_force(env) == 6.0
    # Empty contact sets return the documented zero, without iterating objects.
    data.ncon = 0
    assert current_contact_force(env) == 0.0
