# Copyright (c) Hello Robot, Inc. All rights reserved.

from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
    emet_session_cache_update,
    emet_session_has_current_schema,
    read_emet_robot_id_from_message_or_session,
    read_emet_session,
)


def test_read_emet_robot_id_from_message_or_session() -> None:
    msg = {
        EMET_ZMQ_SESSION_KEY: {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            EMET_ZMQ_ROBOT_ID_KEY: "rby1",
        },
        "step": 0,
    }
    assert read_emet_robot_id_from_message_or_session(msg) == "rby1"


def test_read_emet_session_requires_dict() -> None:
    assert read_emet_session(None) is None
    assert read_emet_session({"emet_session": "bad"}) is None
    inner = {EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION}
    out = read_emet_session({EMET_ZMQ_SESSION_KEY: inner})
    assert out is inner


def test_emet_session_schema_version() -> None:
    assert not emet_session_has_current_schema(None)
    assert not emet_session_has_current_schema({EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: 0})
    assert emet_session_has_current_schema(
        {EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION}
    )


def test_emet_session_cache_update_prefers_higher_step() -> None:
    s1 = {
        EMET_ZMQ_SESSION_KEY: {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "robosuite_sim",
        },
        "step": 1,
    }
    s5 = {
        EMET_ZMQ_SESSION_KEY: {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "robosuite_sim",
            "environment": {"kind": "molmospaces", "scene": "ithor"},
        },
        "step": 5,
    }
    c, st = emet_session_cache_update(None, -1, s1)
    assert st == 1 and c is not None
    c, st = emet_session_cache_update(c, st, s5)
    assert st == 5
    assert c is not None and c.get("environment", {}).get("scene") == "ithor"
