from dataclasses import dataclass


@dataclass
class StarAlignment:
    library_path: str
    exclude_source_id: list[int]
    plot_data_loading: bool

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        return StarAlignment(
            library_path=yaml_dict.get("library_path", ""),
            exclude_source_id=yaml_dict.get("exclude_source_id", []),
            plot_data_loading=yaml_dict.get("plot_data_loading", False),
        )
