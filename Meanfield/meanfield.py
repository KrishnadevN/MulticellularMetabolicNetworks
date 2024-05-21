import numpy as np
from collections import namedtuple
from numba import guvectorize


values = ['Z', 'dZg', 'dZo', 'p', 'Zp', 'ug', 'uo', 'ul']
Vals = namedtuple('Values', values)


@guvectorize(["void(float64[:], float64[:], "
              "float64[:], float64[:], float64[:], "
              "float64[:], float64[:])"],
             '(n),(n)->(n),(n),(n),(n),(n)',
             nopython=True, target='parallel')
def _meanfield(βg, βo, Z, dZg, dZo, φ, p):

    Ug = 1.
    Uo = 3.
    M  = 1.
    K  = 40.

    F  = -3./7.
    F2 = F*F

    βg_orig, βo_orig = βg, βo
    φ[:]  = -0.1
    p[:]  =  0.1
    dugmu, dugbu = np.zeros_like(φ), np.zeros_like(φ)

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

        Kφ    = K*φ
        Kφ3   = -3.*Kφ
        Kφ3o  = Kφ3 - 1./βo

        ugm   = np.maximum(0., M/30. + 7.*Kφ/15.)
        uga   = np.minimum(M/2., Ug)
        ugb   = np.maximum(ugm, Uo/6. + Kφ/2.)

        Ugg   = Ug  - 1./βg
        Uoo   = Uo  - 1./βo
        ugag  = uga - 1./βg
        ugbg  = ugb - 1./βg
        ugb1  = ugb - 1./β1
        ugm1  = ugm - 1./β1
        uga2  = uga - 1./β2
        ugm2  = ugm - 1./β2

        βoKφ3 = βo*Kφ3
        βoKφ3 = np.minimum(200, np.maximum(βoKφ3, -200))

        eKφ3  = np.exp(βoKφ3)/βo
        eguga = np.exp(βg*uga)/βg
        egugb = np.exp(βg*ugb)/βg
        e1ugm = np.exp(β1*ugm)/β1
        e2ugm = np.exp(β2*ugm)/β2
        e1ugb = np.exp(β1*ugb)/β1
        e2uga = np.exp(β2*uga)/β2

        A = Kφ3o  - 6.*(1./β1)
        B = M314o - F *(1./β2)

        Z[:] = (    eKφ3  * (e1ugb - e1ugm) +       eUo * (egUg - egugb)
                -   e314M * (e2uga - e2ugm) -   (1./βo) * (egUg - eguga)    )

        dZg[:] = (  eKφ3  * ( ugb1 * e1ugb - ugm1 * e1ugm )
                +   eUo   * ( Ugg  * egUg  - ugbg * egugb )
                -   e314M * ( uga2 * e2uga - ugm2 * e2ugm )
                - (1./βo) * ( Ugg  * egUg  - ugag * eguga ) )

        dZo[:] = (  eUo   *  (egUg - egugb) * Uoo
                +   i2βo  *  (egUg - eguga)
                +   eKφ3  * ((A + 6.*ugb) * e1ugb - (A + 6.*ugm) * e1ugm )
                -   e314M * ((B + F *uga) * e2uga - (B + F *ugm) * e2ugm )  )

        dugmu[ugm>0.] = 7./15.*K
        dugbu[ugb>0.] = K/2.

        dZu = (     eKφ3  * (-3.*K*βo) * (e1ugb - e1ugm)
                +   eKφ3  * (   dugbu*β1*e1ugb - dugmu*β1*e1ugm )
                +   eUo   * ( - dugbu*βg*egugb )
                -   e314M * ( - dugmu*β2*e2ugm )                    )

        new_φ = -2.*(dZg/Z) + (dZo/Z)/3.
        new_p = -dZu/Z

        dφ, dp = new_φ-φ, new_p-p

        if np.all(np.abs(dφ/φ)) < 1.e-6 and np.all(np.abs(dp/p)) < 1.e-4:
            break

        φ[:]  += 0.001 * dφ
        p[:]  += 0.001 * dp


def meanfield(βg, βo, grid=False):
    if np.squeeze(βg).ndim==1 and np.squeeze(βo).ndim==1:
        if not len(βg) == len(βo):
            grid = True
    else:
        grid = False
    if grid:
        βg, βo = np.broadcast_arrays(βg[:,None], βo[None,:])
    else:
        βg, βo = np.broadcast_arrays(βg, βo)

    Z, dZg, dZo, φ, p = _meanfield(βg, βo)
    Zp = Z * np.exp(p*φ)

    ug = dZg/Z
    uo = dZo/Z
    ul = -2.*ug + uo/3.

    return Vals(Z, dZg, dZo, p, Zp, ug, uo, ul)
