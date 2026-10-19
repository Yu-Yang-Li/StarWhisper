import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.optimize import curve_fit


# basic function for fitting
def zenith(xdata, X, e):
    ca, u = xdata
    b = X + ca
    return np.arccos(np.cos(u) * np.cos(e) - np.cos(b) * np.sin(u) * np.sin(e))


def func_linear(xdata, E):
    return np.sin(xdata + E)


def taylor(xdata, k1, k2, k3, k4):
    return k1 * xdata + k2 * (xdata**3) + k3 * (xdata**5) + k4 * (xdata**7)


def cosx(x, a, b):
    return a * np.cos(x / 180 * np.pi - b)


def sinx(x, c, d):
    return c * np.sin(x / 180 * np.pi - d)


# functions
def bu_from_f(x, y, x0, y0, f):
    # Get projected azimuth and zenith ca(b'), cz(u)
    ca_b = np.arctan2(x - x0, y - y0) + np.pi
    cz_u = 2 * np.arcsin(np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / f)
    # print(ca,cz)
    return ca_b, cz_u


def ini_Ee(ca, cz, az, ze):
    # get initial E and e0
    xdata = [ca, cz]
    ydata = ze
    popt, pcov = curve_fit(zenith, xdata, ydata)
    X, e = popt

    b0 = X + ca
    amE = np.arctan2(
        np.sin(b0) * np.sin(cz),
        (np.cos(b0) * np.sin(cz) * np.cos(e) + np.cos(cz) * np.sin(e)),
    )

    xdata = amE
    ydata = np.sin(az)
    popt, pcov = curve_fit(func_linear, xdata, ydata)

    E = popt[0]
    return E, e


def iteration(x, y, x0, y0, az, ze, E0, e0, mode):
    # first iteration mode
    u = np.arccos(np.cos(ze) * np.cos(e0) + np.sin(ze) * np.sin(e0) * np.cos(az - E0))
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    popt, pcov = curve_fit(taylor, r, u)
    k1, k2, k3, k4 = popt
    cz = taylor(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), k1, k2, k3, k4)
    ca = np.arctan2(x - x0, y - y0) + np.pi

    xdata = [ca, cz]
    ydata = ze
    popt, pcov = curve_fit(zenith, xdata, ydata)
    X, e = popt
    Cze = zenith(xdata, X, e)

    b0 = X + ca
    amE = np.arctan2(
        np.sin(b0) * np.sin(cz),
        (np.cos(b0) * np.sin(cz) * np.cos(e) + np.cos(cz) * np.sin(e)),
    )

    xdata = amE
    ydata = np.sin(az)
    popt, pcov = curve_fit(func_linear, xdata, ydata)
    E = popt[0]
    a0 = X + E
    Caz = np.array([x + E if x + E > 0 else x + E + np.pi * 2 for x in amE])
    if mode == "iter":
        return E, a0, e
    else:
        return Caz, Cze, E, a0, e, k1, k2, k3, k4


def xytoradec(x, y, x0, y0, E, a0, e, k1, k2, k3, k4, time0, loc):
    cz = taylor(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), k1, k2, k3, k4)
    ca = np.arctan2(x - x0, y - y0) + np.pi
    u0 = cz
    X = a0 - E
    b = X + ca
    Cze = np.arccos(np.cos(u0) * np.cos(e) - np.cos(b) * np.sin(u0) * np.sin(e))

    amE = np.arctan2(
        np.sin(b) * np.sin(u0),
        (np.cos(b) * np.sin(u0) * np.cos(e) + np.cos(u0) * np.sin(e)),
    )
    # 处理标量和数组两种情况
    if np.isscalar(amE):
        Caz = amE + E if amE + E > 0 else amE + E + np.pi * 2
    else:
        Caz = np.array([x + E if x + E > 0 else x + E + np.pi * 2 for x in amE])

    Calt = np.pi / 2 - Cze
    time = Time(time0)
    altaz_from_xy = AltAz(obstime=time, location=loc, az=Caz * u.rad, alt=Calt * u.rad)
    skycord_from_xy = SkyCoord(altaz_from_xy)
    radec_from_xy = skycord_from_xy.transform_to("icrs")
    ra = radec_from_xy.ra.degree
    dec = radec_from_xy.dec.degree
    return (
        ra,
        dec,
        Caz * 180 / np.pi,
        Cze * 180 / np.pi,
        u0 * 180 / np.pi,
        ca * 180 / np.pi,
    )


def radectoub(Rra, Rdec, x0, y0, E, a0, e, k1, k2, k3, k4, time0, loc):
    # not in the pipeline, but useful
    time = Time(time0)
    cord = SkyCoord(ra=Rra * u.degree, dec=Rdec * u.degree)
    Raltaz = cord.transform_to(AltAz(obstime=time, location=loc))
    z = np.pi / 2 - Raltaz.alt.rad
    a = Raltaz.az.rad
    u0 = np.arccos(np.cos(z) * np.cos(e) + np.sin(z) * np.sin(e) * np.cos(a - E))
    b0 = np.arctan2(
        np.sin(a - E) * np.sin(z) / np.sin(u0),
        -1 * (np.cos(z) - np.cos(u0) * np.cos(e)) / (np.sin(u0) * np.sin(e)),
    )
    b = np.array([x if x > 0 else x + 2 * np.pi for x in b0])
    ca = np.array(
        [x - a0 + E if x - a0 + E > 0 else x - a0 + E + 2 * np.pi for x in b0]
    )
    return a * 180 / np.pi, z * 180 / np.pi, u0 * 180 / np.pi, ca * 180 / np.pi


def run1(ra1, dec1, x, y, x0, y0, time0, loc, f):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # index=np.array(list(np.where(r<200)[0]))   # skip stars around zenith
    index = np.where(r < 0)[0]  # 直接使用np.where返回的索引数组，确保是整数类型

    ra1 = np.delete(ra1, index)
    dec1 = np.delete(dec1, index)
    x = np.delete(x, index)
    y = np.delete(y, index)
    r = np.delete(r, index)

    # coordinates
    cord = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    time = Time(time0)
    Waltaz = cord.transform_to(AltAz(obstime=time, location=loc))
    # Walt= Waltaz.alt.rad
    Waz = Waltaz.az.rad
    Wze = np.pi / 2 - Waltaz.alt.rad

    ca, cz = bu_from_f(x, y, x0, y0, f)
    E0, e0 = ini_Ee(ca, cz, Waz, Wze)
    return E0, e0, Waz, Wze


def solver(k1, k2, k3, k4, y):
    poly_coeffs = [k4, 0, k3, 0, k2, 0, k1, 0]

    def solve_for_y(poly_coeffs, y):
        pc = poly_coeffs.copy()
        pc[-1] -= y
        return np.roots(pc)

    R = solve_for_y(poly_coeffs, y)
    buffs = []
    for i in R:
        if i.imag == 0:
            r = i.real
            return r
        else:
            continue
        print("solution not exist")
        return 0


def findcenter(ra1, dec1, x, y, x0, y0, time0, loc, f, length, step):
    files_ini = open("findcenter.txt", "w+")
    x00 = x0
    y00 = y0
    X, Y, A, C = [[], [], [], []]
    for k in range(0, length):
        x0 = x00 + (k - length / 2) * step
        for j in range(0, length):
            y0 = y00 + (j - length) * step
            E0, e0, Waz, Wze = run1(ra1, dec1, x, y, x0, y0, time0, loc, f)
            for i in range(20):
                E00 = E0
                E0, a0, e0 = iteration(x, y, x0, y0, Waz, Wze, E0, e0, "iter")
                # print(i+1,'E=%.05f, a0=%.05f, e=%.05f, dE:%.03f%%'%(E0,a0,e0,(E00-E0)/E00*100))
            Caz, Cze, E, a0, e, k1, k2, k3, k4 = iteration(
                x, y, x0, y0, Waz, Wze, E0, e0, ""
            )
            num = -1
            rdif = solver(k1, k2, k3, k4, e0)
            xc = x0 + num * rdif * np.cos(np.pi * 2 - (E - a0) - np.pi / 2)
            yc = y0 + num * rdif * np.sin(np.pi * 2 - (E - a0) - np.pi / 2)
            # print('image center:(%.02f, %0.2f)'%(x0,y0),'sky center:(%.02f, %0.2f)'%(xc,yc))

            c2 = 90 - Cze * 180 / np.pi
            c1 = Caz * 180 / np.pi
            w2 = 90 - Wze * 180 / np.pi
            w1 = Waz * 180 / np.pi

            popt, pcov = curve_fit(cosx, c1, (c2 - w2) * np.pi / 180)
            a, b = popt
            A.append(a)
            popt, pcov = curve_fit(sinx, c1, (c1 - w1) * np.pi / 180)
            c, d = popt
            C.append(c)

            X.append(x0)
            Y.append(y0)
            print(x0, y0, a, c, file=files_ini)
            # print('\r',end='')
            # print("Finding progress: {}%: ".format(j+1+k*length), "▋" * (j+1+k*length // 2), end="")
            # print("Finding progress: %i%%: "%((j+1+k*length)/(length**2)*100), end="")
            print(
                "Finding progress: %i%%: " % ((j + 1 + k * length) / (length**2) * 100)
            )
            sys.stdout.flush()
    return X, Y, A, C
    files_ini.close()


def func2d(xy, a0, a1, a2, a3, a4, a5):  # , a6, a7, a8, a9):
    x, y = xy
    return (
        a0 + a1 * x + a2 * y + a3 * x**2 + a4 * x * y + a5 * y**2
    )  # +a6*x**3 + a7*x**2*y + a8*x*y**2 + a9*y**3


def fitcenter(X, Y, A, C, length, step):
    Aabs = abs(np.array(A))
    Cabs = abs(np.array(C))
    index1 = np.where(Aabs > 1)[0]
    index2 = np.where(Cabs > 1)[0]
    index = np.union1d(index1, index2)  # 使用union1d合并索引，确保是整数类型

    X = np.delete(X, index)
    Y = np.delete(Y, index)
    A = np.delete(A, index)
    C = np.delete(C, index)
    Aabs = np.delete(Aabs, index)
    Cabs = np.delete(Cabs, index)

    popt, pcov = curve_fit(func2d, (X, Y), Cabs)
    X0 = np.linspace(min(X), max(X), int(length / step * 10))
    Y0 = np.linspace(min(Y), max(Y), int(length / step * 10))
    X1, Y1 = np.meshgrid(X0, Y0)
    Z = func2d((X1, Y1), *popt)
    index = np.where(Z == np.min(Z))
    # print(index[0])
    # print(np.min(Z),X1[index[0][0]][index[1][0]],Y1[index[0][0]][index[1][0]],Z[index[0][0]][index[1][0]])
    x0_new = X1[index[0][0]][index[1][0]]
    y0_new = Y1[index[0][0]][index[1][0]]
    print("new image center return")
    return x0_new, y0_new
