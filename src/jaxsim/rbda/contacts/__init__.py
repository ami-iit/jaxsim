from . import relaxed_rigid, rigid, soft, visco_elastic
from .common import ContactModel, ContactsParams
from .relaxed_rigid import RelaxedRigidContacts, RelaxedRigidContactsParams
from .rigid import RigidContacts, RigidContactsParams
from .soft import SoftContacts, SoftContactsParams
from .visco_elastic import ViscoElasticContacts, ViscoElasticContactsParams

ContactParamsTypes = (
    SoftContactsParams
    | RigidContactsParams
    | RelaxedRigidContactsParams
    | ViscoElasticContactsParams
)
