"""ClawTeam 角色智能体模块。"""

from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType
from graph.roles.physician import PhysicianAgent
from graph.roles.pharmacist import PharmacistAgent
from graph.roles.radiologist import RadiologistAgent

__all__ = [
    "RoleAgent",
    "RoleOpinion",
    "RoleType",
    "PhysicianAgent",
    "PharmacistAgent",
    "RadiologistAgent",
]
