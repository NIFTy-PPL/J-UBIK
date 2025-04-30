from dataclasses import dataclass


@dataclass
class FilterData:
    name: str
    filepaths: list[str]


@dataclass
class DataFilePaths:
    filters: list[FilterData]
    step_type: str

    @classmethod
    def from_yaml_dict(cls, yaml: dict):
        step_type = yaml.get("step_type", "tweakregstep")
        filter_files = yaml.get("filter")

        filters = []
        for fltname, paths in filter_files.items():
            filter_file_paths = []
            for path in paths:
                filter_file_paths.append(path.format(step_type=step_type))
            filters.append(FilterData(fltname, filter_file_paths))

        return DataFilePaths(filters=filters, step_type=step_type)
