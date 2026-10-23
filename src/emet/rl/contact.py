"""Safe access to solved MuJoCo contact loads, in newtons."""

import numpy as np


def solved_contact_force(
    model, data, contact_index, *, efc_address=None, dim=None, exclude=0, efc_force=None, friction=None
):
    """Return normal load and an L1 bound on the 3D *linear* contact force.

    Pyramidal contacts store 2*(dim-1) nonnegative facet forces, whose sum
    is the normal load. Elliptic contacts store the contact-frame components
    directly. Decode both without calling the native contact-force extractor.
    Torsional/rolling moments are not added to forces with incompatible units.
    """
    if efc_address is None or dim is None:
        contacts = data.contact
        if contact_index < 0 or contact_index >= len(contacts):
            return 0.0, 0.0
        efc_address = contacts.efc_address[contact_index]
        dim = contacts.dim[contact_index]
        exclude = contacts.exclude[contact_index]
    dim, address = int(dim), int(efc_address)
    pyramidal = int(model.opt.cone) == 0 and dim > 1
    count = 2 * (dim - 1) if pyramidal else dim
    if int(exclude) != 0 or address < 0 or dim not in (1, 3, 4, 6) or address + count > int(data.nefc):
        return 0.0, 0.0
    if efc_force is None:
        efc_force = np.asarray(data.efc_force).copy()
    local = np.asarray(efc_force)[address : address + count]
    if local.size != count or not np.isfinite(local).all():
        return 0.0, 0.0
    if pyramidal:
        if friction is None:
            friction = np.asarray(data.contact.friction[contact_index]).copy()
        friction = np.asarray(friction)
        normal = float(local.sum())
        tangent = friction[:2] * (local[:4:2] - local[1:4:2])
        return normal, normal + float(np.abs(tangent).sum())
    normal = abs(float(local[0]))
    return normal, float(np.abs(local[:3]).sum())
