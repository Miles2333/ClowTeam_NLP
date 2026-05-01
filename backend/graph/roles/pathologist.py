"""病理科医生角色（Tumor Board）。"""

from graph.roles.base_role import RoleAgent, RoleType


class PathologistAgent(RoleAgent):
    role_type = RoleType.PATHOLOGIST
    role_label = "病理科医生"
    prompt_file = "PATHOLOGIST.md"
