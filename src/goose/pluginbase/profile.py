from typing import Any, Dict, List, Mapping, Type

from attrs import asdict, define, field

from .utils import ensure_list


@define
class ToolkitSpec:
    """Configuration for a Toolkit"""

    name: str
    requires: Mapping[str, str] = field(factory=dict)


@define
class Profile:
    """The configuration for a run of goose"""

    provider: str
    processor: str
    accelerator: str
    moderator: str
    toolkits: List[ToolkitSpec] = field(factory=list, converter=ensure_list(ToolkitSpec))

    @toolkits.validator
    def _check_toolkit_requirements(self, _: Type["ToolkitSpec"], toolkits: List[ToolkitSpec]) -> None:
        # checks that the list of toolkits in the profile have their requirements
        installed_toolkits = set([toolkit.name for toolkit in toolkits])

        for toolkit in toolkits:
            toolkit_name = toolkit.name
            toolkit_requirements = toolkit.requires
            for _, req in toolkit_requirements.items():
                if req not in installed_toolkits:
                    msg = f"Toolkit {toolkit_name} requires {req} but it is not present"
                    raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)