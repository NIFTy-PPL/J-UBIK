from dataclasses import dataclass
from configparser import ConfigParser

import numpy as np
import astropy.units as u
import nifty8 as ift

from ..wcs.wcs_astropy import resolve_str_to_quantity


@dataclass
class ResolveSkyModel:
    pass


@dataclass
class RelativePointLocations:
    locations: list[tuple[u.Quantity, u.Quantity]]

    @classmethod
    def cfg_to_relative_locations(cls, locations: str):
        ppos = []
        for xy in locations.split(","):
            x, y = xy.split("$")
            ppos.append((resolve_str_to_quantity(x).to(u.rad),
                        resolve_str_to_quantity(y).to(u.rad)))
        return RelativePointLocations(ppos)

    def to_indices(self, sky_dom: ift.RGSpace, unit: u.Unit = u.rad):
        # FIXME: Change sky_dom to Grid.
        ppos = [(pos[0].to(unit).value, pos[1].to(unit).value)
                for pos in self.locations]
        dx = np.array(sky_dom.distances)
        center = np.array(sky_dom.shape) // 2
        return np.unique(
            np.round(ppos / dx + center).astype(int).T, axis=1)


@dataclass
class ResolvePointSourcesModel:
    locations: RelativePointLocations
    a: float  # alpha
    scale: float  # q

    freq_mode: str = 'single'
    polarization: str = 'I'
    mode: str = 'single'

    @classmethod
    def cfg_to_resolve_point_sources(cls, cfg: ConfigParser):
        freq_mode = cfg.get("freq mode", 'single')
        polarization = cfg.get('polarization', 'I')
        mode = cfg.get('point sources mode', 'single')

        if freq_mode != 'single':
            raise NotImplementedError
        if polarization != 'I':
            raise NotImplementedError
        if mode != 'single':
            raise NotImplementedError

        locations = RelativePointLocations.cfg_to_relative_locations(
            cfg["point sources relative locations"])

        a = cfg.getfloat("point sources a")
        scale = cfg.getfloat("point sources scale")
        if a is None:
            raise ValueError('Provide "a" parameter for point source prior')
        if scale is None:
            raise ValueError(
                'Provide "scale" parameter for point source prior')

        return ResolvePointSourcesModel(
            locations=locations,
            a=a,
            scale=scale,
            freq_mode=freq_mode,
            polarization=polarization,
            mode=mode)

    @classmethod
    def yamldict_to_resolve_point_sources(cls, yaml: dict):
        freq_mode = yaml.get("freq_mode", 'single')
        polarization = yaml.get('polarization', 'I')

        points_cfg = yaml.get('points', {})
        mode = points_cfg.get('mode', 'single')

        if freq_mode != 'single':
            raise NotImplementedError
        if polarization != 'I':
            raise NotImplementedError
        if mode != 'single':
            raise NotImplementedError

        locations = points_cfg['relative_locations']
        raise NotImplementedError
        a = points_cfg['a']
        scale = points_cfg['scale']

        return ResolvePointSourcesModel(
            locations=locations,
            a=a,
            scale=scale,
            freq_mode=freq_mode,
            polarization=polarization,
            mode=mode)
