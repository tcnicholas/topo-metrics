from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import ase
import numpy as np
import numpy.typing as npt

from topo_metrics.knots import (
    writhe_method_1a,
    writhe_method_1b,
    writhe_method_2a,
    writhe_method_2b,
)
from topo_metrics.utils import uniform_repr

if TYPE_CHECKING:
    from topo_metrics.topology import Node


@dataclass(frozen=True)
class RingGeometry:

    nodes: tuple[Node, ...]
    """ The nodes in the ring. """

    @property
    def species(self) -> str:
        """ Species string of the ring. """

        return "".join([str(x.node_type) for x in self.nodes])

    @cached_property
    def positions(self) -> npt.NDArray[np.floating]:
        """ Cartesian positions of the nodes in the ring. """

        return np.asarray([x.cart_coord for x in self.nodes], dtype=float)

    @cached_property
    def radius_of_gyration(self) -> float:
        """ Radius of gyration around the geometric centroid. """

        positions = self.positions
        r_cm = positions.mean(axis=0)
        dr = positions - r_cm
        rg2 = np.mean(np.sum(dr * dr, axis=1))

        return float(np.sqrt(rg2))

    @cached_property
    def gyration_tensor(self) -> npt.NDArray[np.floating]:
        """Gyration tensor of the ring. 
        
        The gyration tensor describes the second moments of position of a set of
        points around their center of mass. It is a symmetric 3x3 matrix.
        """

        positions = self.positions
        r_cm = positions.mean(axis=0)
        dr = positions - r_cm
        Q = (dr.T @ dr) / dr.shape[0]
        return Q

    @cached_property
    def principal_moments(self) -> npt.NDArray[np.floating]:
        """Principal moments of the gyration tensor. 
        
        The principal moments are the eigenvalues of the gyration tensor, which
        describe the distribution of points along the principal axes.
        """

        Q = self.gyration_tensor
        evals, _ = np.linalg.eigh(Q) 
        return evals

    @cached_property
    def asphericity(self) -> float:
        """ Asphericity of the ring based on the principal moments. """

        lam = np.asarray(self.principal_moments, dtype=float)
        s1 = lam.sum()

        if s1 == 0.0:
            return float("nan")

        s2 = np.dot(lam, lam)
        num = s2 - (s1 * s1) / 3.0
        return float(1.5 * num / (s1 * s1))

    def writhe_and_acn(
        self, 
        method: str = "1a",
        closed=True
    ) -> tuple[float, float] | float: 
        """Writhe of the ring using specified method from 

        Parameters
        ----------
        method
            Method to compute writhe. Options are '1a', '1b', '2a', '2b'.
            Default is '1a'. Each method corresponds to those introduced in

        Returns
        -------
        Writhe of the ring.
        """

        P = self.positions
    
        if method == "1a":
            return writhe_method_1a(P, closed=closed)
        elif method == "1b":
            return writhe_method_1b(P, closed=closed)
        elif method == "2a":
            return writhe_method_2a(P)
        elif method == "2b":
            return writhe_method_2b(P)
        else:
            raise ValueError(f"Unknown writhe method: {method}")

    @property
    def geometric_centroid(self) -> npt.NDArray[np.floating]:
        """ Geometric centroid of the ring. """

        return self.positions.mean(axis=0)

    def to_xyz(self, filename: Path | str, write_info: bool = False) -> None:
        """ Write the ring to an xyz file. """

        filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(exist_ok=True, parents=True)

        atoms = ase.Atoms(self.species, positions=self.positions)
        atoms.write(filename)

    def __len__(self) -> int:
        """ The number of nodes in the ring. """

        return len(self.nodes)

    def __repr__(self) -> str:
        """ Tidy string representation of the RingGeometry. """
    
        info = {"n": len(self.nodes)}
        return uniform_repr(
            "RingGeometry",
            **info,
            stringify=True,
            indent_size=4
        )