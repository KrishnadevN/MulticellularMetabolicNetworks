import numpy as np
from collections import namedtuple
from numba import njit, guvectorize, prange



partition_values  = ['Z', 'dZg', 'dZo', 'ddZg', 'ddZo', 'ddZgo',
                     'Zplus', 'dZplusg', 'dZpluso', 'Zp', 'p']
calculated_values = ['ug', 'uo', 'ul', 'dug', 'duo', 'dul',
                     'exchange', 'fraction']

PartVals = namedtuple('PartitionValues', partition_values)
CalcVals = namedtuple('CalculatedValues', calculated_values)



def match_arrays(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.ndim(a)==1 and np.ndim(b)==1:
        pass
    elif np.ndim(a)==0 and np.ndim(b)==1:
        a = np.broadcast_to(a, b.shape)
    elif np.ndim(a)==1 and np.ndim(b)==0:
        b = np.broadcast_to(b, a.shape)
    else:
        raise ValueError('Check dimensions of input arrays')
    if len(a)==len(b):
        pass
    else:
        raise ValueError('Input arrays have incompatible lengths')
    return a, b


@njit
def _meanfield_single(βg, βo):

    Ug = 1.
    Uo = 3.
    M  = 1.
    K  = 25.

    F  = -3./7.
    F2 = F*F

    βg_orig, βo_orig = βg, βo
    ul = -0.1
    p  =  0.1

    for _ in range(100000):

        βg = βg_orig + 2.*p
        βo = βo_orig - p/3.

        β1 = βg + 6.*βo
        β2 = βg + F *βo

        M314  = M * 3./14.
        M314o = M314 - 1./βo
        i2βg  = 1./(βg*βg)
        i2βo  = 1./(βo*βo)
        i2β1  = 1./(β1*β1)
        i2β2  = 1./(β2*β2)

        e314M = np.exp(M314*βo)/βo
        eUo   = np.exp(βo*Uo)/βo
        egUg  = np.exp(βg*Ug)/βg

        Kul   = K*ul
        Kul3  = -3.*Kul
        Kul3o = Kul3 - 1./βo

        ugm   = max(0., M/30. + 7.*Kul/15.)
        uga   = min(M/2., Ug)
        ugb   = max(ugm, Uo/6. + Kul/2.)

        Ugg   = Ug  - 1./βg
        Uoo   = Uo  - 1./βo
        ugag  = uga - 1./βg
        ugbg  = ugb - 1./βg
        ugb1  = ugb - 1./β1
        ugm1  = ugm - 1./β1
        uga2  = uga - 1./β2
        ugm2  = ugm - 1./β2

        # Prevent overflow
        βoKul3 = βo*Kul3
        βoKul3 = min(200, max(βoKul3, -200))

        eKul3 = np.exp(βoKul3)/βo
        eguga = np.exp(βg*uga)/βg
        egugb = np.exp(βg*ugb)/βg
        e1ugm = np.exp(β1*ugm)/β1
        e2ugm = np.exp(β2*ugm)/β2
        e1ugb = np.exp(β1*ugb)/β1
        e2uga = np.exp(β2*uga)/β2

        A = Kul3o - 6.*(1./β1)
        B = M314o - F *(1./β2)

        Z = (       eKul3 * (e1ugb - e1ugm) +       eUo * (egUg - egugb)
                -   e314M * (e2uga - e2ugm) -   (1./βo) * (egUg - eguga)    )

        dZg = (     eKul3 * ( ugb1 * e1ugb - ugm1 * e1ugm )
                +   eUo   * ( Ugg  * egUg  - ugbg * egugb )
                -   e314M * ( uga2 * e2uga - ugm2 * e2ugm )
                - (1./βo) * ( Ugg  * egUg  - ugag * eguga ) )

        dZo = (     eUo   *  (egUg - egugb) * Uoo
                +   i2βo  *  (egUg - eguga)
                +   eKul3 * ((A + 6.*ugb) * e1ugb - (A + 6.*ugm) * e1ugm )
                -   e314M * ((B + F *uga) * e2uga - (B + F *ugm) * e2ugm )  )

        dugmu = 7./15.*K if ugm > 0. else 0.
        dugbu = K/2. if ugb > 0. else 0.

        dZu = (     eKul3 * (-3.*K*βo) * (e1ugb - e1ugm)
                +   eKul3 * (   dugbu*β1*e1ugb - dugmu*β1*e1ugm )
                +   eUo   * ( - dugbu*βg*egugb )
                -   e314M * ( - dugmu*β2*e2ugm )                    )

        ug = dZg/Z
        uo = dZo/Z

        new_ul = -2.*ug + uo/3.
        new_p  = -dZu/Z

        dul, dp = new_ul-ul, new_p-p

        if abs(dul/ul) < 1.e-6 and abs(dp/p) < 1.e-6:
            break

        ul += 0.001 * dul
        p  += 0.001 * dp

        Zp = Z * np.exp(p*ul)

    ddZg = (    eKul3 * (( i2β1 + ugb1**2 ) * e1ugb
                -        ( i2β1 + ugm1**2 ) * e1ugm )
            +   eUo   * (( i2βg + Ugg **2 ) * egUg
                -        ( i2βg + ugbg**2 ) * egugb )
            -   e314M * (( i2β2 + uga2**2 ) * e2uga
                -        ( i2β2 + ugm2**2 ) * e2ugm )
            - (1./βo) * (( i2βg + Ugg **2 ) * egUg
                -        ( i2βg + ugag**2 ) * eguga ) )

    ddZo = (    eUo * (i2βo + Uoo**2) * (egUg - egugb)
            -   2./(βo*βo*βo)         * (egUg - eguga)
            + eKul3 * ( ( i2βo + 36.*i2β1 + (A + 6.*ugb)**2 ) * e1ugb
                      - ( i2βo + 36.*i2β1 + (A + 6.*ugm)**2 ) * e1ugm )
            - e314M * ( ( i2βo + F2 *i2β2 + (B + F *uga)**2 ) * e2uga
                      - ( i2βo + F2 *i2β2 + (B + F *ugm)**2 ) * e2ugm ) )

    ddZgo = (       eUo   *     ( Ugg * egUg - ugbg * egugb )  * Uoo
                +   i2βo  *     ( Ugg * egUg - ugag * eguga )
                +   eKul3 * ( ( 6.*i2β1 + (A + 6.*ugb) * ugb1 ) * e1ugb
                            - ( 6.*i2β1 + (A + 6.*ugm) * ugm1 ) * e1ugm )
                -   e314M * ( ( F *i2β2 + (B + F *uga) * uga2 ) * e2uga
                            - ( F *i2β2 + (B + F *ugm) * ugm2 ) * e2ugm ) )

    # For exchange

    eKul31  =  eKul3 - 1./βo

    Zplus   =  eKul31 * (e1ugb - e1ugm)
    dZplusg =  eKul31 * (ugb1 * e1ugb - ugm1 * e1ugm )
    dZpluso = ( eKul3 * ((   Kul3o + 6.*ugb1 ) * e1ugb
                    -    (   Kul3o + 6.*ugm1 ) * e1ugm )
              - 1./βo * (( - 1./βo + 6.*ugb1 ) * e1ugb
                    -    ( - 1./βo + 6.*ugm1 ) * e1ugm ) )

    return Z, dZg, dZo, ddZg, ddZo, ddZgo, Zplus, dZplusg, dZpluso, Zp, p



@njit
def meanfield_single(βg, βo):

    part_vals = PartVals(*_meanfield_single(βg, βo))
    Z, dZg, dZo, ddZg, ddZo, ddZgo, Zplus, dZplusg, dZpluso, Zp, p = part_vals

    ug = dZg/Z
    uo = dZo/Z
    ul = -2.*ug + uo/3.

    dug = np.sqrt(ddZg/Z - ug*ug)
    duo = np.sqrt(ddZo/Z - uo*uo)
    dul = np.sqrt( (-2.*dug)**2 + (duo/3.)**2 - (4./3.)*(ddZgo/Z - ug*uo) )

    exchange = (-2.*dZplusg + dZpluso/3.)/Z
    fraction = Zplus/Z

    calc_vals = CalcVals(ug, uo, ul, dug, duo, dul, exchange, fraction)

    return calc_vals, part_vals



@guvectorize(["void(float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:])"],
             '(n),(n)->(n),(n),(n),(n),(n),(n),(n),(n)',
             nopython=True, target='parallel')
def _meanfield_calc(βg, βo, ug, uo, ul, dug, duo, dul,
                    exchange, fraction):
    items = βg.shape[0]
    for item in prange(items):
        calc_vals, _ = meanfield_single(βg[item], βo[item])
        arrays = ug, uo, ul, dug, duo, dul, exchange, fraction
        for i in range(len(arrays)):
            (arrays[i])[item] = calc_vals[i]


def meanfield_calc(βg, βo):
    βg, βo = match_arrays(βg, βo)
    return CalcVals(*_meanfield_calc(βg, βo))


meanfield = meanfield_calc


@guvectorize(["void(float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:])"],
             '(n),(n)->(n),(n),(n),(n),(n),(n),(n),(n),(n),(n),(n)',
             nopython=True, target='parallel')
def _meanfield_part(βg, βo, Z, dZg, dZo, ddZg, ddZo, ddZgo,
                   Zplus, dZplusg, dZpluso, Zp, p):
    items = βg.shape[0]
    for item in prange(items):
        part_vals = _meanfield_single(βg[item], βo[item])
        arrays = Z, dZg, dZo, ddZg, ddZo, ddZgo, Zplus, dZplusg, dZpluso, Zp, p
        for i in range(len(arrays)):
            (arrays[i])[item] = part_vals[i]


def meanfield_part(βg, βo):
    βg, βo = match_arrays(βg, βo)
    return PartVals(*_meanfield_part(βg, βo))


@guvectorize(["void(float64[:], float64[:], int64, float64[:,:])"],
             '(m),(n),()->(m,n)', nopython=True, target='parallel')
def _meanfield_grid_calc(βo, βg, item, grid):
    num_βo, num_βg = len(βo), len(βg)
    for βo_count in prange(num_βo):
        for βg_count in prange(num_βg):
            set_1, set_2 = meanfield_single(βg[βg_count], βo[βo_count])
            grid[βo_count, βg_count] = set_1[item]


@guvectorize(["void(float64[:], float64[:], int64, float64[:,:])"],
             '(m),(n),()->(m,n)', nopython=True, target='parallel')
def _meanfield_grid_part(βo, βg, item, grid):
    num_βo, num_βg = len(βo), len(βg)
    for βo_count in prange(num_βo):
        for βg_count in prange(num_βg):
            set_1, set_2 = meanfield_single(βg[βg_count], βo[βo_count])
            grid[βo_count, βg_count] = set_2[item]


def meanfield_grid(βo, βg, value):
    all_vals = [*calculated_values, *partition_values]
    try:
        item = all_vals.index(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'.\n"
                         f"Allowed values: {all_vals}")
    set_id = 1 * (item >= len(calculated_values))
    index  = item - (set_id * len(calculated_values))
    if set_id == 0:
        return _meanfield_grid_calc(βo, βg, index)
    else:
        return _meanfield_grid_part(βo, βg, index)

