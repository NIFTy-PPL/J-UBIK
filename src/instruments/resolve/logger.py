# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society


def _logger_init():
    import logging
    res = logging.getLogger('resolve')
    res.setLevel(logging.WARNING)
    res.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    res.addHandler(ch)
    return res


logger = _logger_init()
