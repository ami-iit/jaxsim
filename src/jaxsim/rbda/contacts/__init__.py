from . import relaxed_rigid, rigid, soft
from .common import ContactModel, ContactsParams
from .relaxed_rigid import RelaxedRigidContacts, RelaxedRigidContactsParams
from .rigid import RigidContacts, RigidContactsParams
from .soft import SoftContacts, SoftContactsParams

ContactParamsTypes = (
    SoftContactsParams | RigidContactsParams | RelaxedRigidContactsParams
)
