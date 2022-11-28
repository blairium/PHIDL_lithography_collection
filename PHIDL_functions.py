"""
This collection of functions was created by Blair Haydon
b.haydon@latrobe.edu.au
https://github.com/blairium/PHIDL_lithography_collection

To use at to working directory and import
"""


import phidl.geometry as pg
from phidl import Device, Layer, LayerSet
from phidl import quickplot as qp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def siemens_star(r, min_feature=2, num_spikes=128, unit=1, layer=0):
    """Produces a Siemens star
    Parameters
    ----------
    r : int or float
        radius of Siemens star
    cut : int or float
        min feature size
    num_spikes : int,
        number of spikes in star
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.
    Returns
    -------
    D : Device
        A Device containing the grating
        geometry.
    """

    theta = np.arange(-np.pi, np.pi, 2 * np.pi / num_spikes)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    D = Device()
    for k in range(0, int(num_spikes / 2)):
        D.add_polygon(
            [(0, 0), (x[2 * k - 1], y[2 * k - 1]), (x[2 * k], y[2 * k])], layer=layer
        )

    r2 = (num_spikes * min_feature * unit) / (2 * np.pi)
    circ = pg.circle(radius=r2, angle_resolution=r2 / 10, layer=0)
    C = pg.boolean(
        A=D, B=circ, operation="not", precision=1e-6, num_divisions=[1, 1], layer=0
    )
    return C


def fit2wafer(wafer, die, diameter, diex, diey, spacing=0):
    """Packs die in regular spacing on a wafer.

    Parameters
    ----------
    die : Device
        The die to fit to a wafer.
    diameter : int or float
        Wafer diameter
    diex : int or float
        die width
    diey : int or float
        die height
    spacing: int or float
        dead space for dicing etc
    Returns
    -------
    D : Device
        A Device containing as many die as possible
    """
    D = Device()

    for x in np.arange(-diameter / 2, diameter / 2, diex + (spacing / 2)):
        for y in np.arange(-diameter / 2, diameter / 2, diey + (spacing / 2)):
            if x**2 + y**2 < (diameter / 2) ** 2:
                D.add_ref(die).movex(x - ((diex / 2) + (spacing / 2))).movey(
                    y - ((diey / 2) + (spacing / 2))
                )
            else:
                pass
    wafer.add_ref(D)
    return wafer


def FZP(D, wl, delta_r, layer):

    """
    Paramaters
    D: Lens diameter
    wl: wavelength
    delta_r: Outzone thickness
    ----------
    Returns
    D: Device
    """

    f = (D * delta_r) / wl  # approx
    num_zones = D / (4 * delta_r)

    n_raddii = np.zeros(int(num_zones))

    for n in range(1, int(num_zones)):
        n_raddii[-n] = np.sqrt(n * wl * f)

    D = pg.circle(radius=n_raddii[-1], angle_resolution=2.5, layer=layer)
    for r in range(1, int(num_zones)):
        if (r % 2) == 0:
            D.add_ref(
                pg.ring(
                    radius=n_raddii[-r],
                    width=n_raddii[-r] - n_raddii[-r - 1],
                    angle_resolution=2.5e-3,
                    layer=0,
                )
            )
    return D


def gc(radius, initial_angle, final_angle, points=500):
    """
    This methods generates points in a circle shape at (0,0) with a specific radius and from a
    starting angle to a final angle.
    Args:
        radius: radius of the circle in microns
        initial_angle: initial angle of the drawing in degrees
        final_angle: final angle of the drawing in degrees
        points: amount of points to be generated (default 199)
    Returns:
        Set of points that form the circle
    """
    theta = np.linspace(np.deg2rad(initial_angle), np.deg2rad(final_angle), points)

    return radius * np.cos(theta), radius * np.sin(theta)


def wafer(size, deadspace=False):
    """
    This methods generates points in a notched wafer with a diameter specified in mm
    Args:
        size: Wafer diameter in mm
        deadspace: if True returns layer with 10 mm deadspace.
    Returns:
        Notched wafer device.
    """
    D = Device(f"{size} Wafer")
    D.add_polygon([gc((size * 1e3 / 2), 0 - 71.03, 180 + 71.03)], layer=1)
    if deadspace == True:
        useable_area = D.add_polygon(
            [gc(((size * 1e3 / 2) - 10e3), 0 - 71.03, 180 + 71.03)], layer=2
        )
    return D


def PEC_test(min_feature, layer=0):
    """
    Copies the proximity effect test pattern found here
    https://ebeam.wnf.uw.edu/ebeamweb/doc/patternprep/patternprep/proximity_main.html
    Parameters
    ----------
    min_feature: Minimum feature size

    Returns
    -------
    D : Device
        A Device containing the grating
        geometry.

    """
    D = Device("PEC test")

    size = int(min_feature * 5)

    i = -1

    for x in np.arange(0, size + 1, min_feature / 2):
        i += 1
        j = -1
        for y in np.arange(0, size + 1, min_feature / 2):
            j += 1
            if (i % 2) == 0:
                D.add_ref(pg.rectangle(size=(size, min_feature), layer=layer)).movey(
                    x * 2
                )  # First set of horizontal lines
                if (j % 2) == 0:
                    D.add_ref(
                        pg.rectangle(size=(min_feature, min_feature), layer=layer)
                    ).movex(size + 2 * x).movey(
                        2 * (size - y)
                    )  # Even set of squares
            elif (i % 2 and j % 2) == 1:
                D.add_ref(
                    pg.rectangle(size=(min_feature, min_feature), layer=layer)
                ).movex(size + 2 * x).movey(
                    2 * (size - y)
                )  # Odd Sqares
                D.add_ref(
                    pg.rectangle(size=((3 * size - 2 * x), min_feature), layer=layer)
                ).movex((size * 3) + min_feature).movey(
                    2 * x
                )  # Second (odd) set of lines
                D.add_ref(
                    pg.rectangle(size=(min_feature, (3 * size - 2 * x)), layer=layer)
                ).movex((2 * (3 * size - x)) + min_feature).movey(
                    2 * x
                )  # Corresponding vertical lines
                if i > 1:
                    D.add_ref(
                        pg.rectangle(size=(min_feature, 2 * size), layer=layer)
                    ).movex((2 * (3 * size - x)) + 2 * min_feature).movey(3 * size)

    D.add_ref(pg.rectangle(size=(min_feature, 4 * size), layer=layer)).movex(
        ((7 * size - 2 * x)) + min_feature
    ).movey(5 * size)
    PEC = pg.offset(
        D, distance=0, join_first=True, precision=1e-6, num_divisions=[1, 1], layer=0
    )
    return PEC
