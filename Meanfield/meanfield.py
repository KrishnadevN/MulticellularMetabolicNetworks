import numpy as np
from collections import namedtuple
from numba import guvectorize, prange


values = ['Z', 'dZg', 'dZo', 'p', 'Zp', 'ug', 'uo', 'ul']
Vals = namedtuple('Values', values)


@guvectorize(["void(float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:])"],
             '(n),(n)->(n),(n),(n),(n),(n)',
             nopython=True, target='parallel')
def _meanfield(βg_vals, βo_vals, Z, dZg, dZo, φ, p):

    Ug = 1.
    Uo = 3.
    M  = 1.
    K  = 40.

    F  = -3./7.
    F2 = F*F

    for i in prange(βg_vals.shape[0]):

        φ[i] = -0.1
        p[i] =  0.1

        for _ in range(100000):

            βg = βg_vals[i] + 2.*p[i]
            βo = βo_vals[i] - p[i]/3.

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

            Kφ    = K*φ[i]
            Kφ3   = -3.*Kφ
            Kφ3o  = Kφ3 - 1./βo

            ugm   = max(0., M/30. + 7.*Kφ/15.)
            uga   = min(M/2., Ug)
            ugb   = max(ugm, Uo/6. + Kφ/2.)

            Ugg   = Ug  - 1./βg
            Uoo   = Uo  - 1./βo
            ugag  = uga - 1./βg
            ugbg  = ugb - 1./βg
            ugb1  = ugb - 1./β1
            ugm1  = ugm - 1./β1
            uga2  = uga - 1./β2
            ugm2  = ugm - 1./β2

            βoKφ3 = βo*Kφ3
            βoKφ3 = min(200, max(βoKφ3, -200))

            eKφ3  = np.exp(βoKφ3)/βo
            eguga = np.exp(βg*uga)/βg
            egugb = np.exp(βg*ugb)/βg
            e1ugm = np.exp(β1*ugm)/β1
            e2ugm = np.exp(β2*ugm)/β2
            e1ugb = np.exp(β1*ugb)/β1
            e2uga = np.exp(β2*uga)/β2

            A = Kφ3o  - 6.*(1./β1)
            B = M314o - F *(1./β2)

            Z[i] = (    eKφ3  * (e1ugb - e1ugm) +       eUo * (egUg - egugb)
                    -   e314M * (e2uga - e2ugm) -   (1./βo) * (egUg - eguga)    )

            dZg[i] = (  eKφ3  * ( ugb1 * e1ugb - ugm1 * e1ugm )
                    +   eUo   * ( Ugg  * egUg  - ugbg * egugb )
                    -   e314M * ( uga2 * e2uga - ugm2 * e2ugm )
                    - (1./βo) * ( Ugg  * egUg  - ugag * eguga ) )

            dZo[i] = (  eUo   *  (egUg - egugb) * Uoo
                    +   i2βo  *  (egUg - eguga)
                    +   eKφ3  * ((A + 6.*ugb) * e1ugb - (A + 6.*ugm) * e1ugm )
                    -   e314M * ((B + F *uga) * e2uga - (B + F *ugm) * e2ugm )  )

            dugmu = 7./15.*K if ugm > 0. else 0.
            dugbu = K/2. if ugb > 0. else 0.

            dZu = (     eKφ3  * (-3.*K*βo) * (e1ugb - e1ugm)
                    +   eKφ3  * (   dugbu*β1*e1ugb - dugmu*β1*e1ugm )
                    +   eUo   * ( - dugbu*βg*egugb )
                    -   e314M * ( - dugmu*β2*e2ugm )                    )

            new_φ = -2.*(dZg[i]/Z[i]) + (dZo[i]/Z[i])/3.
            new_p = -dZu/Z[i]

            dφ, dp = new_φ-φ[i], new_p-p[i]

            if np.abs(dφ) < 1.e-4 and np.abs(dp) < 1.e-4:
                break

            φ[i] += 0.001 * dφ
            p[i] += 0.001 * dp


def meanfield(βg, βo, grid=False):
    """
    Compute the mean-field solution for the given parameter values βg and βo.

    Parameters
    ----------
    βg : float or array-like
        The parameter value corresponding to glucose
    βo : float or array-like
        The parameter value corresponding to oxygen
    grid : bool, default=False
        Whether to compute the solution on a grid of βg and βo values
        If both βg and βo are one-dimensional arrays of non-matching lengths,
        the solution will be automatically computed on a grid.

    Returns
    -------
    Vals : A named tuple containing the following fields.
           (All fields are either floats or arrays of the same shape as βg and βo.)
        Z   : The partition function
        dZg : Derivative of the partition function with respect to βg
        dZo : Derivative of the partition function with respect to βo
        p   : Order parameter p
        Zp  : The partition function Z multiplied by
              the exponential of the product of the order parameters
        ug  : Glucose flux
        uo  : Oxygen flux
        ul  : Lactate flux
    """
    if np.squeeze(βg).ndim==1 and np.squeeze(βo).ndim==1:
        if not len(βg)==len(βo):
            grid = True
    else:
        grid = False
    βg, βo = np.atleast_1d(βg), np.atleast_1d(βo)
    if grid:
        βg, βo = np.broadcast_arrays(βg[:,None], βo[None,:])
    else:
        βg, βo = np.broadcast_arrays(βg, βo)

    Z, dZg, dZo, φ, p = _meanfield(βg, βo)

    if np.squeeze(Z).ndim==0:
        Z, dZg, dZo, φ, p = (value.item() for value in values)

    Zp = Z * np.exp(p*φ)

    ug = dZg/Z
    uo = dZo/Z
    ul = -2.*ug + uo/3.

    return Vals(Z, dZg, dZo, p, Zp, ug, uo, ul)
