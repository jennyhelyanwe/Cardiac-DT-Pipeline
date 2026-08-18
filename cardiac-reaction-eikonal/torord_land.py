"""
ToRORd-Land electromechanical model — Python conversion.

Original MATLAB source: model_ToRORd_Land.m, modelRunner.m,
getCurrentsStructure.m, getStartingState.m

Reference: Tomek et al. (2019) ToR-ORd, Land et al. (2017) contraction model.

Usage example
-------------
from torod_land import get_starting_state, model_runner, get_currents_structure

X0 = get_starting_state('m_endo')
parameters = {'bcl': 1000, 'model': model_ToRORd_Land}
time, X, parameters = model_runner(X0, parameters, beats=200, ignore_first=195)
currents = get_currents_structure(time, X, parameters, ignore_first_spikes=0)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(params: dict, key: str, default):
    return params.get(key, default)


# ---------------------------------------------------------------------------
# Sub-functions (ionic currents, fluxes)
# ---------------------------------------------------------------------------

def get_INa_Grandi(v, m, h, hp, j, jp, fINap, ENa, INa_Multiplier):
    # m gate
    mss = 1.0 / (1.0 + np.exp(-(56.86 + v) / 9.03))**2
    taum = (0.1292 * np.exp(-((v + 45.79) / 15.54)**2)
            + 0.06487 * np.exp(-((v - 4.823) / 51.12)**2))
    dm = (mss - m) / taum

    # h gate
    if v >= -40:
        ah = 0.0
        bh = 0.77 / (0.13 * (1.0 + np.exp(-(v + 10.66) / 11.1)))
    else:
        ah = 0.057 * np.exp(-(v + 80) / 6.8)
        bh = 2.7 * np.exp(0.079 * v) + 3.1e5 * np.exp(0.3485 * v)
    tauh = 1.0 / (ah + bh)
    hss = 1.0 / (1.0 + np.exp((v + 71.55) / 7.43))**2
    dh = (hss - h) / tauh

    # j gate
    if v >= -40:
        aj = 0.0
        bj = (0.6 * np.exp(0.057 * v)) / (1.0 + np.exp(-0.1 * (v + 32)))
    else:
        aj = ((-2.5428e4 * np.exp(0.2444 * v) - 6.948e-6 * np.exp(-0.04391 * v))
              * (v + 37.78)) / (1.0 + np.exp(0.311 * (v + 79.23)))
        bj = (0.02424 * np.exp(-0.01052 * v)) / (1.0 + np.exp(-0.1378 * (v + 40.14)))
    tauj = 1.0 / (aj + bj)
    jss = hss
    dj = (jss - j) / tauj

    # phosphorylated gates
    hssp = 1.0 / (1.0 + np.exp((v + 71.55 + 6) / 7.43))**2
    dhp = (hssp - hp) / tauh
    taujp = 1.46 * tauj
    djp = (jss - jp) / taujp

    GNa = 11.7802
    INa = INa_Multiplier * GNa * (v - ENa) * m**3 * ((1.0 - fINap) * h * j + fINap * hp * jp)
    return INa, dm, dh, dhp, dj, djp


def get_INaL(v, mL, hL, hLp, fINaLp, ENa, celltype, INaL_Multiplier):
    mLss = 1.0 / (1.0 + np.exp(-(v + 42.85) / 5.264))
    tmL = (0.1292 * np.exp(-((v + 45.79) / 15.54)**2)
           + 0.06487 * np.exp(-((v - 4.823) / 51.12)**2))
    dmL = (mLss - mL) / tmL
    hLss = 1.0 / (1.0 + np.exp((v + 87.61) / 7.488))
    thL = 200.0
    dhL = (hLss - hL) / thL
    hLssp = 1.0 / (1.0 + np.exp((v + 93.81) / 7.488))
    thLp = 3.0 * thL
    dhLp = (hLssp - hLp) / thLp
    GNaL = 0.0279 * INaL_Multiplier
    if celltype == 1:
        GNaL *= 0.6
    INaL = GNaL * (v - ENa) * mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)
    return INaL, dmL, dhL, dhLp


def get_Ito(v, a, iF, iS, ap, iFp, iSp, fItop, EK, celltype, Ito_Multiplier):
    ass_ = 1.0 / (1.0 + np.exp(-(v - 14.34) / 14.82))
    ta = 1.0515 / (1.0 / (1.2089 * (1.0 + np.exp(-(v - 18.4099) / 29.3814)))
                   + 3.5 / (1.0 + np.exp((v + 100.0) / 29.3814)))
    da = (ass_ - a) / ta
    iss = 1.0 / (1.0 + np.exp((v + 43.94) / 5.711))
    if celltype == 1:
        delta_epi = 1.0 - (0.95 / (1.0 + np.exp((v + 70.0) / 5.0)))
    else:
        delta_epi = 1.0
    tiF = (4.562 + 1.0 / (0.3933 * np.exp(-(v + 100.0) / 100.0)
                           + 0.08004 * np.exp((v + 50.0) / 16.59)))
    tiS = (23.62 + 1.0 / (0.001416 * np.exp(-(v + 96.52) / 59.05)
                           + 1.780e-8 * np.exp((v + 114.1) / 8.079)))
    tiF *= delta_epi
    tiS *= delta_epi
    AiF = 1.0 / (1.0 + np.exp((v - 213.6) / 151.2))
    AiS = 1.0 - AiF
    diF = (iss - iF) / tiF
    diS = (iss - iS) / tiS
    i_ = AiF * iF + AiS * iS
    assp = 1.0 / (1.0 + np.exp(-(v - 24.34) / 14.82))
    dap = (assp - ap) / ta
    dti_develop = 1.354 + 1.0e-4 / (np.exp((v - 167.4) / 15.89) + np.exp(-(v - 12.23) / 0.2154))
    dti_recover = 1.0 - 0.5 / (1.0 + np.exp((v + 70.0) / 20.0))
    tiFp = dti_develop * dti_recover * tiF
    tiSp = dti_develop * dti_recover * tiS
    diFp = (iss - iFp) / tiFp
    diSp = (iss - iSp) / tiSp
    ip = AiF * iFp + AiS * iSp
    Gto = 0.16 * Ito_Multiplier
    if celltype in (1, 2):
        Gto *= 2.0
    Ito = Gto * (v - EK) * ((1.0 - fItop) * a * i_ + fItop * ap * ip)
    return Ito, da, diF, diS, dap, diFp, diSp


def get_ICaL(v, d, ff, fs, fcaf, fcas, jca, nca, nca_i, ffp, fcafp,
             fICaLp, cai, cass, cao, nai, nass, nao, ki, kss, ko,
             cli, clo, celltype, ICaL_fractionSS, ICaL_PCaMultiplier):
    R, T, F = 8314.0, 310.0, 96485.0
    vffrt = v * F * F / (R * T)
    vfrt = v * F / (R * T)

    # d gate
    dss = 1.0763 * np.exp(-1.0070 * np.exp(-0.0829 * v))
    if v > 31.4978:
        dss = 1.0
    td = 0.6 + 1.0 / (np.exp(-0.05 * (v + 6.0)) + np.exp(0.09 * (v + 14.0)))
    dd = (dss - d) / td

    fss = 1.0 / (1.0 + np.exp((v + 19.58) / 3.696))
    tff = 7.0 + 1.0 / (0.0045 * np.exp(-(v + 20.0) / 10.0) + 0.0045 * np.exp((v + 20.0) / 10.0))
    tfs = 1000.0 + 1.0 / (0.000035 * np.exp(-(v + 5.0) / 4.0) + 0.000035 * np.exp((v + 5.0) / 6.0))
    Aff = 0.6
    Afs = 1.0 - Aff
    dff = (fss - ff) / tff
    dfs = (fss - fs) / tfs
    f_ = Aff * ff + Afs * fs

    fcass = fss
    tfcaf = 7.0 + 1.0 / (0.04 * np.exp(-(v - 4.0) / 7.0) + 0.04 * np.exp((v - 4.0) / 7.0))
    tfcas = 100.0 + 1.0 / (0.00012 * np.exp(-v / 3.0) + 0.00012 * np.exp(v / 7.0))
    Afcaf = 0.3 + 0.6 / (1.0 + np.exp((v - 10.0) / 10.0))
    Afcas = 1.0 - Afcaf
    dfcaf = (fcass - fcaf) / tfcaf
    dfcas = (fcass - fcas) / tfcas
    fca = Afcaf * fcaf + Afcas * fcas

    tjca = 75.0
    jcass = 1.0 / (1.0 + np.exp((v + 18.08) / 2.7916))
    djca = (jcass - jca) / tjca

    tffp = 2.5 * tff
    dffp = (fss - ffp) / tffp
    fp_ = Aff * ffp + Afs * fs

    tfcafp = 2.5 * tfcaf
    dfcafp = (fcass - fcafp) / tfcafp
    fcap = Afcaf * fcafp + Afcas * fcas

    # SS nca
    Kmn = 0.002
    k2n = 500.0
    km2n = jca * 1.0
    anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass)**4.0)
    dnca = anca * k2n - nca * km2n

    # Myo nca
    anca_i = 1.0 / (k2n / km2n + (1.0 + Kmn / cai)**4.0)
    dnca_i = anca_i * k2n - nca_i * km2n

    def _activity_coefficients(nai_, ki_, cai_, nao_, ko_, cao_, cli, clo):
        Io = 0.5 * (nao_ + ko_ + clo + 4 * cao_) / 1000.0
        Ii = 0.5 * (nai_ + ki_ + cli + 4 * cai_) / 1000.0
        dielConstant = 74.0
        temp = 310.0
        constA = 1.82e6 * (dielConstant * temp)**(-1.5)
        def _g(z, I):
            return np.exp(-constA * z**2 * (np.sqrt(I) / (1 + np.sqrt(I)) - 0.3 * I))
        gamma_cai = _g(2, Ii)
        gamma_cao = _g(2, Io)
        gamma_nai = _g(1, Ii)
        gamma_nao = _g(1, Io)
        gamma_ki  = _g(1, Ii)
        gamma_kao = _g(1, Io)
        return gamma_cai, gamma_cao, gamma_nai, gamma_nao, gamma_ki, gamma_kao

    # SS driving forces
    gc_cai_ss, gc_cao_ss, gc_nai_ss, gc_nao_ss, gc_ki_ss, gc_kao_ss = \
        _activity_coefficients(nass, kss, cass, nao, ko, cao, cli, clo)
    PhiCaL_ss  = 4.0 * vffrt * (gc_cai_ss * cass * np.exp(2.0 * vfrt) - gc_cao_ss * cao) / (np.exp(2.0 * vfrt) - 1.0)
    PhiCaNa_ss = 1.0 * vffrt * (gc_nai_ss * nass * np.exp(vfrt) - gc_nao_ss * nao) / (np.exp(vfrt) - 1.0)
    PhiCaK_ss  = 1.0 * vffrt * (gc_ki_ss  * kss  * np.exp(vfrt) - gc_kao_ss * ko)  / (np.exp(vfrt) - 1.0)

    # Myo driving forces
    gc_cai_i, gc_cao_i, gc_nai_i, gc_nao_i, gc_ki_i, gc_kao_i = \
        _activity_coefficients(nai, ki, cai, nao, ko, cao, cli, clo)
    gammaCaoMyo = gc_cao_i
    gammaCaiMyo = gc_cai_i
    PhiCaL_i  = 4.0 * vffrt * (gc_cai_i * cai * np.exp(2.0 * vfrt) - gc_cao_i * cao) / (np.exp(2.0 * vfrt) - 1.0)
    PhiCaNa_i = 1.0 * vffrt * (gc_nai_i * nai * np.exp(vfrt) - gc_nao_i * nao) / (np.exp(vfrt) - 1.0)
    PhiCaK_i  = 1.0 * vffrt * (gc_ki_i  * ki  * np.exp(vfrt) - gc_kao_i * ko)  / (np.exp(vfrt) - 1.0)

    PCa = 8.3757e-5 * ICaL_PCaMultiplier
    if celltype == 1:
        PCa *= 1.2
    elif celltype == 2:
        PCa *= 1.8
    PCap   = 1.1 * PCa
    PCaNa  = 0.00125 * PCa
    PCaK   = 3.574e-4 * PCa
    PCaNap = 0.00125 * PCap
    PCaKp  = 3.574e-4 * PCap

    def _gate(nca_): return f_ * (1.0 - nca_) + jca * fca * nca_
    def _gatep(nca_): return fp_ * (1.0 - nca_) + jca * fcap * nca_

    ICaL_ss  = ((1.0 - fICaLp) * PCa  * PhiCaL_ss  * d * _gate(nca)
                + fICaLp        * PCap * PhiCaL_ss  * d * _gatep(nca))
    ICaNa_ss = ((1.0 - fICaLp) * PCaNa  * PhiCaNa_ss * d * _gate(nca)
                + fICaLp        * PCaNap * PhiCaNa_ss * d * _gatep(nca))
    ICaK_ss  = ((1.0 - fICaLp) * PCaK   * PhiCaK_ss  * d * _gate(nca)
                + fICaLp        * PCaKp  * PhiCaK_ss  * d * _gatep(nca))

    ICaL_i   = ((1.0 - fICaLp) * PCa  * PhiCaL_i  * d * _gate(nca_i)
                + fICaLp        * PCap * PhiCaL_i  * d * _gatep(nca_i))
    ICaNa_i  = ((1.0 - fICaLp) * PCaNa  * PhiCaNa_i * d * _gate(nca_i)
                + fICaLp        * PCaNap * PhiCaNa_i * d * _gatep(nca_i))
    ICaK_i   = ((1.0 - fICaLp) * PCaK   * PhiCaK_i  * d * _gate(nca_i)
                + fICaLp        * PCaKp  * PhiCaK_i  * d * _gatep(nca_i))

    ICaL_i   *= (1.0 - ICaL_fractionSS)
    ICaNa_i  *= (1.0 - ICaL_fractionSS)
    ICaK_i   *= (1.0 - ICaL_fractionSS)
    ICaL_ss  *= ICaL_fractionSS
    ICaNa_ss *= ICaL_fractionSS
    ICaK_ss  *= ICaL_fractionSS

    return (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i,
            dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp, dfcafp,
            PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo)


def get_IKr_MM(V, c0, c1, c2, o, i_, ko, EK, celltype, IKr_Multiplier):
    R, T, F = 8314.0, 310.0, 96485.0
    vfrt = V * F / (R * T)
    alpha  = 0.1161 * np.exp(0.2990 * vfrt)
    beta   = 0.2442 * np.exp(-1.604 * vfrt)
    alpha1 = 1.25 * 0.1235
    beta1  = 0.1911
    alpha2 = 0.0578 * np.exp(0.9710 * vfrt)
    beta2  = 0.349e-3 * np.exp(-1.062 * vfrt)
    alphai = 0.2533 * np.exp(0.5953 * vfrt)
    betai  = 1.25 * 0.0522 * np.exp(-0.8209 * vfrt)
    alphac2ToI = 0.52e-4 * np.exp(1.525 * vfrt)
    betaItoC2  = beta2 * betai * alphac2ToI / (alpha2 * alphai)

    dc0 = c1 * beta - c0 * alpha
    dc1 = c0 * alpha + c2 * beta1 - c1 * (beta + alpha1)
    dc2 = c1 * alpha1 + o * beta2 + i_ * betaItoC2 - c2 * (beta1 + alpha2 + alphac2ToI)
    do  = c2 * alpha2 + i_ * betai - o * (beta2 + alphai)
    di  = c2 * alphac2ToI + o * alphai - i_ * (betaItoC2 + betai)

    GKr = 0.0321 * np.sqrt(ko / 5.0) * IKr_Multiplier
    if celltype == 1:
        GKr *= 1.3
    elif celltype == 2:
        GKr *= 0.8
    IKr = GKr * o * (V - EK)
    return IKr, dc0, dc1, dc2, do, di


def get_IKs(v, xs1, xs2, cai, EKs, celltype, IKs_Multiplier):
    xs1ss = 1.0 / (1.0 + np.exp(-(v + 11.60) / 8.932))
    txs1  = 817.3 + 1.0 / (2.326e-4 * np.exp((v + 48.28) / 17.80)
                            + 0.001292 * np.exp(-(v + 210.0) / 230.0))
    dxs1  = (xs1ss - xs1) / txs1
    xs2ss = xs1ss
    txs2  = 1.0 / (0.01 * np.exp((v - 50.0) / 20.0) + 0.0193 * np.exp(-(v + 66.54) / 31.0))
    dxs2  = (xs2ss - xs2) / txs2
    KsCa  = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai)**1.4)
    GKs   = 0.0011 * IKs_Multiplier
    if celltype == 1:
        GKs *= 1.4
    IKs = GKs * KsCa * xs1 * xs2 * (v - EKs)
    return IKs, dxs1, dxs2


def get_IK1(v, ko, EK, celltype, IK1_Multiplier):
    aK1  = 4.094 / (1.0 + np.exp(0.1217 * (v - EK - 49.934)))
    bK1  = (15.72 * np.exp(0.0674 * (v - EK - 3.257))
            + np.exp(0.0618 * (v - EK - 594.31))) / (1.0 + np.exp(-0.1629 * (v - EK + 14.207)))
    K1ss = aK1 / (aK1 + bK1)
    GK1  = IK1_Multiplier * 0.6992
    if celltype == 1:
        GK1 *= 1.2
    elif celltype == 2:
        GK1 *= 1.3
    IK1 = GK1 * np.sqrt(ko / 5.0) * K1ss * (v - EK)
    return IK1


def get_INaCa(v, F, R, T, nass, nai, nao, cass, cai, cao,
              celltype, INaCa_Multiplier, INaCa_fractionSS):
    zca = 2.0
    kna1, kna2, kna3 = 15.0, 5.0, 88.12
    kasymm = 12.5
    wna, wca, wnaca = 6.0e4, 6.0e4, 5.0e3
    kcaon, kcaoff = 1.5e6, 5.0e3
    qna, qca = 0.5224, 0.1670
    hca = np.exp(qca * v * F / (R * T))
    hna = np.exp(qna * v * F / (R * T))

    def _ncx_cycle(na_i, ca_i):
        h1  = 1 + na_i / kna3 * (1 + hna)
        h2  = na_i * hna / (kna3 * h1)
        h3  = 1.0 / h1
        h4  = 1.0 + na_i / kna1 * (1 + na_i / kna2)
        h5  = na_i * na_i / (h4 * kna1 * kna2)
        h6  = 1.0 / h4
        h7  = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
        h8  = nao / (kna3 * hna * h7)
        h9  = 1.0 / h7
        h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
        h11 = nao * nao / (h10 * kna1 * kna2)
        h12 = 1.0 / h10
        k1  = h12 * cao * kcaon
        k2  = kcaoff
        k3p_  = h9 * wca
        k3pp_ = h8 * wnaca
        k3  = k3p_ + k3pp_
        k4p_  = h3 * wca / hca
        k4pp_ = h2 * wnaca
        k4  = k4p_ + k4pp_
        k5  = kcaoff
        k6  = h6 * ca_i * kcaon
        k7  = h5 * h2 * wna
        k8  = h8 * h11 * wna
        x1  = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2  = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3  = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4  = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        denom = x1 + x2 + x3 + x4
        E1, E2, E3, E4 = x1/denom, x2/denom, x3/denom, x4/denom
        allo = 1.0 / (1.0 + (150.0e-6 / ca_i)**2)
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp_ - E2 * k3pp_
        JncxCa = E2 * k2 - E1 * k1
        return allo, JncxNa, JncxCa

    Gncx = 0.0034 * INaCa_Multiplier
    if celltype == 1:
        Gncx *= 1.1
    elif celltype == 2:
        Gncx *= 1.4

    allo_i, JncxNa_i, JncxCa_i = _ncx_cycle(nai, cai)
    INaCa_i  = (1.0 - INaCa_fractionSS) * Gncx * allo_i * (1.0 * JncxNa_i + zca * JncxCa_i)

    allo_ss, JncxNa_ss, JncxCa_ss = _ncx_cycle(nass, cass)
    INaCa_ss = INaCa_fractionSS * Gncx * allo_ss * (1.0 * JncxNa_ss + zca * JncxCa_ss)

    return INaCa_i, INaCa_ss


def get_INaK(v, F, R, T, nai, nao, ki, ko, celltype, INaK_Multiplier):
    k1p, k1m = 949.5, 182.4
    k2p, k2m = 687.2, 39.4
    k3p, k3m = 1899.0, 79300.0
    k4p, k4m = 639.0, 40.0
    Knai0, Knao0 = 9.073, 27.78
    delta = -0.1550
    Knai = Knai0 * np.exp(delta * v * F / (3.0 * R * T))
    Knao = Knao0 * np.exp((1.0 - delta) * v * F / (3.0 * R * T))
    Kki, Kko = 0.5, 0.3582
    MgADP, MgATP = 0.05, 9.8
    Kmgatp = 1.698e-7
    H = 1.0e-7
    eP = 4.2
    Khp, Knap, Kxkur = 1.698e-7, 224.0, 292.0
    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)
    a1 = (k1p * (nai / Knai)**3) / ((1.0 + nai / Knai)**3 + (1.0 + ki / Kki)**2 - 1.0)
    b1 = k1m * MgADP
    a2 = k2p
    b2 = (k2m * (nao / Knao)**3) / ((1.0 + nao / Knao)**3 + (1.0 + ko / Kko)**2 - 1.0)
    a3 = (k3p * (ko / Kko)**2) / ((1.0 + nao / Knao)**3 + (1.0 + ko / Kko)**2 - 1.0)
    b3 = k3m * P * H / (1.0 + MgATP / Kmgatp)
    a4 = k4p * MgATP / Kmgatp / (1.0 + MgATP / Kmgatp)
    b4 = (k4m * (ki / Kki)**2) / ((1.0 + nai / Knai)**3 + (1.0 + ki / Kki)**2 - 1.0)
    x1 = a4*a1*a2 + b2*b4*b3 + a2*b4*b3 + b3*a1*a2
    x2 = b2*b1*b4 + a1*a2*a3 + a3*b1*b4 + a2*a3*b4
    x3 = a2*a3*a4 + b3*b2*b1 + b2*b1*a4 + a3*a4*b1
    x4 = b4*b3*b2 + a3*a4*a1 + b2*a4*a1 + b3*b2*a1
    denom = x1 + x2 + x3 + x4
    E1, E2, E3, E4 = x1/denom, x2/denom, x3/denom, x4/denom
    JnakNa = 3.0 * (E1 * a3 - E2 * b3)
    JnakK  = 2.0 * (E4 * b1 - E3 * a1)
    Pnak = 15.4509 * INaK_Multiplier
    if celltype == 1:
        Pnak *= 0.9
    elif celltype == 2:
        Pnak *= 0.7
    INaK = Pnak * (1.0 * JnakNa + 1.0 * JnakK)
    return INaK


def get_Jrel(Jrelnp, Jrelp, ICaL, cass, cajsr, fJrelp, celltype, Jrel_Multiplier):
    jsrMidpoint = 1.7
    bt = 4.75
    a_rel = 0.5 * bt
    Jrel_inf = a_rel * (-ICaL) / (1.0 + (jsrMidpoint / cajsr)**8)
    if celltype == 2:
        Jrel_inf *= 1.7
    tau_rel = bt / (1.0 + 0.0123 / cajsr)
    tau_rel = max(tau_rel, 0.001)
    dJrelnp = (Jrel_inf - Jrelnp) / tau_rel

    btp = 1.25 * bt
    a_relp = 0.5 * btp
    Jrel_infp = a_relp * (-ICaL) / (1.0 + (jsrMidpoint / cajsr)**8)
    if celltype == 2:
        Jrel_infp *= 1.7
    tau_relp = btp / (1.0 + 0.0123 / cajsr)
    tau_relp = max(tau_relp, 0.001)
    dJrelp = (Jrel_infp - Jrelp) / tau_relp

    Jrel = Jrel_Multiplier * 1.5378 * ((1.0 - fJrelp) * Jrelnp + fJrelp * Jrelp)
    return Jrel, dJrelnp, dJrelp


def get_Jup(cai, cansr, fJupp, celltype, Jup_Multiplier):
    Jupnp = Jup_Multiplier * 0.005425 * cai / (cai + 0.00092)
    Jupp  = Jup_Multiplier * 2.75 * 0.005425 * cai / (cai + 0.00092 - 0.00017)
    if celltype == 1:
        Jupnp *= 1.3
        Jupp  *= 1.3
    Jleak = Jup_Multiplier * 0.0048825 * cansr / 15.0
    Jup   = (1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak
    return Jup, Jleak


# ---------------------------------------------------------------------------
# Main ODE function
# ---------------------------------------------------------------------------

def model_ToRORd_Land(t, X, flag_ode,
                      cellType=0,
                      ICaL_Multiplier=1, INa_Multiplier=1, Ito_Multiplier=1,
                      INaL_Multiplier=1, IKr_Multiplier=1, IKs_Multiplier=1,
                      IK1_Multiplier=1, IKb_Multiplier=1, INaCa_Multiplier=1,
                      INaK_Multiplier=1, INab_Multiplier=1, ICab_Multiplier=1,
                      IpCa_Multiplier=1, ICaCl_Multiplier=1, IClb_Multiplier=1,
                      Jrel_Multiplier=1, Jup_Multiplier=1,
                      nao=140, cao=1.8, ko=5,
                      ICaL_fractionSS=0.8, INaCa_fractionSS=0.35,
                      stimAmp=-53, stimDur=1,
                      vcParameters=None, apClamp=None, extraParams=None):

    celltype = cellType

    # --- unpack state vector ---
    v       = X[0]
    nai     = X[1];  nass    = X[2]
    ki      = X[3];  kss     = X[4]
    cai     = X[5];  cass    = X[6]
    cansr   = X[7];  cajsr   = X[8]
    m       = X[9];  hp      = X[10]; h       = X[11]; j_gate  = X[12]
    jp      = X[13]; mL      = X[14]; hL      = X[15]; hLp     = X[16]
    a       = X[17]; iF      = X[18]; iS      = X[19]; ap      = X[20]
    iFp     = X[21]; iSp     = X[22]
    d       = X[23]; ff      = X[24]; fs      = X[25]; fcaf    = X[26]
    fcas    = X[27]; jca     = X[28]; nca     = X[29]; nca_i   = X[30]
    ffp     = X[31]; fcafp   = X[32]
    xs1     = X[33]; xs2     = X[34]
    Jrel_np = X[35]; CaMKt   = X[36]
    ikr_c0  = X[37]; ikr_c1  = X[38]; ikr_c2  = X[39]
    ikr_o   = X[40]; ikr_i   = X[41]
    Jrel_p  = X[42]

    cli = 24.0
    clo = 150.0

    # Land-Niederer state variables
    XS        = max(0.0, X[43])
    XW        = max(0.0, X[44])
    Ca_TRPN   = max(0.0, X[45])
    TmBlocked = X[46]
    ZETAS     = X[47]
    ZETAW     = X[48]

    # --- Land-Niederer contraction model ---
    mode = 'intact'
    lam  = 1.0
    lambda_rate = 0.0

    perm50 = 0.35; TRPN_n = 2; koff = 0.1
    dr = 0.25; wfrac = 0.5; TOT_A = 25
    ktm_unblock = 0.021
    beta_1 = -2.4; beta_0 = 2.3
    gamma_xb = 0.0085; gamma_wu = 0.615; phi = 2.23

    nperm = 2.036; ca50 = 0.805; Tref = 120; nu = 7; mu = 3

    k_ws = 0.004 * mu
    k_uw = 0.026 * nu

    lambda_min = 0.87; lambda_max = 1.2

    cdw = phi * k_uw * (1 - dr) * (1 - wfrac) / ((1 - dr) * wfrac)
    cds = phi * k_ws * (1 - dr) * wfrac / dr
    k_wu = k_uw * (1.0 / wfrac - 1) - k_ws
    k_su = k_ws * (1.0 / dr - 1) * wfrac
    A    = (0.25 * TOT_A) / ((1 - dr) * wfrac + dr) * (dr / 0.25)

    lambda0 = min(lambda_max, lam)
    Lfac    = max(0.0, 1 + beta_0 * (lambda0 + min(lambda_min, lambda0) - (1 + lambda_min)))

    XU = (1 - TmBlocked) - XW - XS
    xb_ws = k_ws * XW
    xb_uw = k_uw * XU
    xb_wu = k_wu * XW
    xb_su = k_su * XS

    gamma_rate   = gamma_xb * max((ZETAS > 0) * ZETAS, (ZETAS < -1) * (-ZETAS - 1))
    xb_su_gamma  = gamma_rate * XS
    gamma_rate_w = gamma_wu * abs(ZETAW)
    xb_wu_gamma  = gamma_rate_w * XW

    dXS       = xb_ws - xb_su - xb_su_gamma
    dXW       = xb_uw - xb_wu - xb_ws - xb_wu_gamma
    ca50_eff  = ca50 + beta_1 * min(0.2, lam - 1)
    dCa_TRPN  = koff * ((cai * 1000 / ca50_eff)**TRPN_n * (1 - Ca_TRPN) - Ca_TRPN)

    XSSS = dr * 0.5
    XWSS = (1 - dr) * wfrac * 0.5
    ktm_block = ktm_unblock * (perm50**nperm) * 0.5 / (0.5 - XSSS - XWSS)
    dTmBlocked = (ktm_block * min(100.0, Ca_TRPN**(-(nperm / 2))) * XU
                  - ktm_unblock * Ca_TRPN**(nperm / 2) * TmBlocked)

    dZETAS = A * lambda_rate - cds * ZETAS
    dZETAW = A * lambda_rate - cdw * ZETAW

    Ta = Lfac * (Tref / dr) * ((ZETAS + 1) * XS + ZETAW * XW)

    # --- physical constants ---
    R_gas = 8314.0; T_body = 310.0; F_farad = 96485.0

    # --- cell geometry ---
    L_cell = 0.01; rad = 0.0011
    vcell  = 1000 * 3.14159 * rad**2 * L_cell
    Ageo   = 2 * 3.14159 * rad**2 + 2 * 3.14159 * rad * L_cell
    Acap   = 2 * Ageo
    vmyo   = 0.68 * vcell; vnsr = 0.0552 * vcell
    vjsr   = 0.0048 * vcell; vss = 0.02 * vcell

    # --- CaMK ---
    KmCaMK  = 0.15; aCaMK = 0.05; bCaMK = 0.00068
    CaMKo   = 0.05; KmCaM = 0.0015
    CaMKb   = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa   = CaMKb + CaMKt
    dCaMKt  = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt

    # --- reversal potentials ---
    ENa  = (R_gas * T_body / F_farad) * np.log(nao / nai)
    EK   = (R_gas * T_body / F_farad) * np.log(ko / ki)
    PKNa = 0.01833
    EKs  = (R_gas * T_body / F_farad) * np.log((ko + PKNa * nao) / (ki + PKNa * nai))

    vffrt = v * F_farad**2 / (R_gas * T_body)
    vfrt  = v * F_farad / (R_gas * T_body)

    # --- CaMK phosphorylation fractions ---
    fINap  = 1.0 / (1.0 + KmCaMK / CaMKa)
    fINaLp = fINap
    fItop  = fINap
    fICaLp = fINap

    # --- ionic currents ---
    INa, dm, dh, dhp, dj_gate, djp = get_INa_Grandi(
        v, m, h, hp, j_gate, jp, fINap, ENa, INa_Multiplier)

    INaL, dmL, dhL, dhLp = get_INaL(
        v, mL, hL, hLp, fINaLp, ENa, celltype, INaL_Multiplier)

    Ito, da, diF, diS, dap, diFp, diSp = get_Ito(
        v, a, iF, iS, ap, iFp, iSp, fItop, EK, celltype, Ito_Multiplier)

    (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i,
     dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp, dfcafp,
     PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo) = get_ICaL(
        v, d, ff, fs, fcaf, fcas, jca, nca, nca_i, ffp, fcafp,
        fICaLp, cai, cass, cao, nai, nass, nao, ki, kss, ko,
        cli, clo, celltype, ICaL_fractionSS, ICaL_Multiplier)

    ICaL     = ICaL_ss  + ICaL_i
    ICaNa    = ICaNa_ss + ICaNa_i
    ICaK     = ICaK_ss  + ICaK_i
    ICaL_tot = ICaL + ICaNa + ICaK

    IKr, dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o, dt_ikr_i = get_IKr_MM(
        v, ikr_c0, ikr_c1, ikr_c2, ikr_o, ikr_i, ko, EK, celltype, IKr_Multiplier)

    IKs, dxs1, dxs2 = get_IKs(v, xs1, xs2, cai, EKs, celltype, IKs_Multiplier)

    IK1 = get_IK1(v, ko, EK, celltype, IK1_Multiplier)

    INaCa_i, INaCa_ss = get_INaCa(
        v, F_farad, R_gas, T_body, nass, nai, nao, cass, cai, cao,
        celltype, INaCa_Multiplier, INaCa_fractionSS)

    INaK = get_INaK(v, F_farad, R_gas, T_body, nai, nao, ki, ko, celltype, INaK_Multiplier)

    # --- background / minor currents ---
    xkb = 1.0 / (1.0 + np.exp(-(v - 10.8968) / 23.9871))
    GKb = 0.0189 * IKb_Multiplier
    if celltype == 1:
        GKb *= 0.6
    IKb = GKb * xkb * (v - EK)

    PNab = 1.9239e-9 * INab_Multiplier
    INab = PNab * vffrt * (nai * np.exp(vfrt) - nao) / (np.exp(vfrt) - 1.0)

    PCab = 5.9194e-8 * ICab_Multiplier
    ICab = PCab * 4.0 * vffrt * (gammaCaiMyo * cai * np.exp(2.0 * vfrt) - gammaCaoMyo * cao) / (np.exp(2.0 * vfrt) - 1.0)

    GpCa = 5e-4 * IpCa_Multiplier
    IpCa = GpCa * cai / (0.0005 + cai)

    # --- chloride currents ---
    ecl = (R_gas * T_body / F_farad) * np.log(cli / clo)
    Fjunc = 1.0; Fsl = 1.0 - Fjunc
    GClCa = ICaCl_Multiplier * 0.2843
    GClB  = IClb_Multiplier  * 1.98e-3
    KdClCa = 0.1
    I_ClCa_junc = Fjunc * GClCa / (1 + KdClCa / cass) * (v - ecl)
    I_ClCa_sl   = Fsl   * GClCa / (1 + KdClCa / cai)  * (v - ecl)
    I_ClCa = I_ClCa_junc + I_ClCa_sl
    I_Clbk = GClB * (v - ecl)

    # --- calcium handling ---
    fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
    Jrel, dJrel_np, dJrel_p = get_Jrel(
        Jrel_np, Jrel_p, ICaL_ss, cass, cajsr, fJrelp, celltype, Jrel_Multiplier)

    fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
    Jup, Jleak = get_Jup(cai, cansr, fJupp, celltype, Jup_Multiplier)

    Jtr = (cansr - cajsr) / 60.0

    # --- stimulus ---
    Istim = stimAmp if t <= stimDur else 0.0

    # --- membrane voltage ---
    dv = -(INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1
            + INaCa_i + INaCa_ss + INaK + INab + IKb + IpCa + ICab
            + I_ClCa + I_Clbk + Istim)

    # --- diffusion fluxes ---
    JdiffNa = (nass - nai) / 2.0
    JdiffK  = (kss  - ki)  / 2.0
    Jdiff   = (cass - cai) / 0.2

    # --- concentration ODEs ---
    dnai  = (-(ICaNa_i + INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab)
              * Acap / (F_farad * vmyo) + JdiffNa * vss / vmyo)
    dnass = (-(ICaNa_ss + 3.0 * INaCa_ss)
              * Acap / (F_farad * vss) - JdiffNa)

    dki   = (-(ICaK_i + Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK)
              * Acap / (F_farad * vmyo) + JdiffK * vss / vmyo)
    dkss  = -ICaK_ss * Acap / (F_farad * vss) - JdiffK

    cmdnmax = 0.05 * (1.3 if celltype == 1 else 1.0)
    kmcmdn  = 0.00238
    trpnmax = 0.07
    Bcai    = 1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + cai)**2)
    dcai    = Bcai * (-(ICaL_i + IpCa + ICab - 2.0 * INaCa_i) * Acap / (2.0 * F_farad * vmyo)
                       - Jup * vnsr / vmyo + Jdiff * vss / vmyo - dCa_TRPN * trpnmax)

    BSRmax, KmBSR = 0.047, 0.00087
    BSLmax, KmBSL = 1.124, 0.0087
    Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + cass)**2
                       + BSLmax * KmBSL / (KmBSL + cass)**2)
    dcass = Bcass * (-(ICaL_ss - 2.0 * INaCa_ss) * Acap / (2.0 * F_farad * vss)
                      + Jrel * vjsr / vss - Jdiff)

    dcansr = Jup - Jtr * vjsr / vnsr

    csqnmax, kmcsqn = 10.0, 0.8
    Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr)**2)
    dcajsr = Bcajsr * (Jtr - Jrel)

    if flag_ode == 1:
        return np.array([
            dv, dnai, dnass, dki, dkss, dcai, dcass, dcansr, dcajsr,
            dm, dhp, dh, dj_gate, djp, dmL, dhL, dhLp,
            da, diF, diS, dap, diFp, diSp,
            dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp, dfcafp,
            dxs1, dxs2, dJrel_np, dCaMKt,
            dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o, dt_ikr_i,
            dJrel_p, dXS, dXW, dCa_TRPN, dTmBlocked, dZETAS, dZETAW
        ])
    else:
        return np.array([
            INa, INaL, Ito, ICaL, IKr, IKs, IK1, INaCa_i, INaCa_ss, INaK,
            IKb, INab, ICab, IpCa, Jdiff, JdiffNa, JdiffK, Jup, Jleak, Jtr,
            Jrel, CaMKa, Istim,
            fINap, fINaLp, fICaLp, fJrelp, fJupp,
            cajsr, cansr, PhiCaL_ss, v, Ta, lam
        ])


# ---------------------------------------------------------------------------
# modelRunner
# ---------------------------------------------------------------------------

def model_runner(X0, parameters: dict, beats: int, ignore_first: int):
    """
    Run the ToRORd-Land model for `beats` beats, returning the last
    (beats - ignore_first) beats.

    Parameters
    ----------
    X0 : array-like, length 49
    parameters : dict  (keys: bcl, model, cellType, multipliers, etc.)
    beats : int
    ignore_first : int

    Returns
    -------
    time : list of 1-D arrays
    X    : list of 2-D arrays  (rows = time points, cols = state variables)
    parameters : dict (updated, with isFailed flag if ODE failed)
    """
    cellType          = _p(parameters, 'cellType', 0)
    verbose           = _p(parameters, 'verbose', False)
    nao               = _p(parameters, 'nao', 140)
    cao               = _p(parameters, 'cao', 1.8)
    ko                = _p(parameters, 'ko', 5)
    ICaL_fractionSS   = _p(parameters, 'ICaL_fractionSS', 0.8)
    INaCa_fractionSS  = _p(parameters, 'INaCa_fractionSS', 0.35)
    INa_Multiplier    = _p(parameters, 'INa_Multiplier', 1)
    ICaL_Multiplier   = _p(parameters, 'ICaL_Multiplier', 1)
    Ito_Multiplier    = _p(parameters, 'Ito_Multiplier', 1)
    INaL_Multiplier   = _p(parameters, 'INaL_Multiplier', 1)
    IKr_Multiplier    = _p(parameters, 'IKr_Multiplier', 1)
    IKs_Multiplier    = _p(parameters, 'IKs_Multiplier', 1)
    IK1_Multiplier    = _p(parameters, 'IK1_Multiplier', 1)
    IKb_Multiplier    = _p(parameters, 'IKb_Multiplier', 1)
    INaCa_Multiplier  = _p(parameters, 'INaCa_Multiplier', 1)
    INaK_Multiplier   = _p(parameters, 'INaK_Multiplier', 1)
    INab_Multiplier   = _p(parameters, 'INab_Multiplier', 1)
    ICab_Multiplier   = _p(parameters, 'ICab_Multiplier', 1)
    IpCa_Multiplier   = _p(parameters, 'IpCa_Multiplier', 1)
    ICaCl_Multiplier  = _p(parameters, 'ICaCl_Multiplier', 1)
    IClb_Multiplier   = _p(parameters, 'IClb_Multiplier', 1)
    Jrel_Multiplier   = _p(parameters, 'Jrel_Multiplier', 1)
    Jup_Multiplier    = _p(parameters, 'Jup_Multiplier', 1)
    stimAmp           = _p(parameters, 'stimAmp', -53)
    stimDur           = _p(parameters, 'stimDur', 1)
    CL                = parameters['bcl']
    model_fn          = _p(parameters, 'model', model_ToRORd_Land)

    extra_args = (
        cellType,
        ICaL_Multiplier, INa_Multiplier, Ito_Multiplier, INaL_Multiplier,
        IKr_Multiplier, IKs_Multiplier, IK1_Multiplier, IKb_Multiplier,
        INaCa_Multiplier, INaK_Multiplier, INab_Multiplier, ICab_Multiplier,
        IpCa_Multiplier, ICaCl_Multiplier, IClb_Multiplier,
        Jrel_Multiplier, Jup_Multiplier,
        nao, cao, ko, ICaL_fractionSS, INaCa_fractionSS,
        stimAmp, stimDur,
        None, None, None   # vcParameters, apClamp, extraParams
    )

    def ode_rhs(t, y):
        return model_fn(t, y, 1, *extra_args)

    time_list = []
    X_list    = []
    X0_now    = np.array(X0, dtype=float)

    from scipy.integrate import solve_ivp

    for n in range(beats):
        if verbose:
            print(f'Beat = {n + 1}')
        sol = solve_ivp(ode_rhs, [0.0, CL], X0_now,
                        method='Radau',
                        rtol=1e-6, atol=1e-8,
                        dense_output=False)
        if not sol.success:
            parameters['isFailed'] = 1
            # trim already-kept beats
            time_list = time_list[ignore_first:]
            X_list    = X_list[ignore_first:]
            return time_list, X_list, parameters

        time_list.append(sol.t)
        X_list.append(sol.y.T)   # shape (n_points, n_states)
        X0_now = sol.y[:, -1]

    time_list = time_list[ignore_first:]
    X_list    = X_list[ignore_first:]
    return time_list, X_list, parameters


# ---------------------------------------------------------------------------
# getCurrentsStructure
# ---------------------------------------------------------------------------

def get_currents_structure(time: list, X: list, parameters: dict, ignore_first_spikes: int = 0):
    """
    Re-evaluate all currents and fluxes from stored state trajectories.

    Parameters
    ----------
    time : list of 1-D arrays  (output of model_runner, already trimmed)
    X    : list of 2-D arrays
    parameters : dict
    ignore_first_spikes : int  (additional beats to skip from the front of time/X)

    Returns
    -------
    currents : dict of 1-D arrays
    """
    cellType         = _p(parameters, 'cellType', 0)
    nao              = _p(parameters, 'nao', 140)
    cao              = _p(parameters, 'cao', 1.8)
    ko               = _p(parameters, 'ko', 5)
    ICaL_fractionSS  = _p(parameters, 'ICaL_fractionSS', 0.8)
    INaCa_fractionSS = _p(parameters, 'INaCa_fractionSS', 0.24)
    INa_Multiplier   = _p(parameters, 'INa_Multiplier', 1)
    ICaL_Multiplier  = _p(parameters, 'ICaL_Multiplier', 1)
    Ito_Multiplier   = _p(parameters, 'Ito_Multiplier', 1)
    INaL_Multiplier  = _p(parameters, 'INaL_Multiplier', 1)
    IKr_Multiplier   = _p(parameters, 'IKr_Multiplier', 1)
    IKs_Multiplier   = _p(parameters, 'IKs_Multiplier', 1)
    IK1_Multiplier   = _p(parameters, 'IK1_Multiplier', 1)
    IKb_Multiplier   = _p(parameters, 'IKb_Multiplier', 1)
    INaCa_Multiplier = _p(parameters, 'INaCa_Multiplier', 1)
    INaK_Multiplier  = _p(parameters, 'INaK_Multiplier', 1)
    INab_Multiplier  = _p(parameters, 'INab_Multiplier', 1)
    ICab_Multiplier  = _p(parameters, 'ICab_Multiplier', 1)
    IpCa_Multiplier  = _p(parameters, 'IpCa_Multiplier', 1)
    ICaCl_Multiplier = _p(parameters, 'ICaCl_Multiplier', 1)
    IClb_Multiplier  = _p(parameters, 'IClb_Multiplier', 1)
    Jrel_Multiplier  = _p(parameters, 'Jrel_Multiplier', 1)
    Jup_Multiplier   = _p(parameters, 'Jup_Multiplier', 1)
    stimAmp          = _p(parameters, 'stimAmp', -53)
    stimDur          = _p(parameters, 'stimDur', 1)
    bcl              = parameters.get('bcl', 1000)
    model_fn         = _p(parameters, 'model', model_ToRORd_Land)

    extra_args = (
        cellType,
        ICaL_Multiplier, INa_Multiplier, Ito_Multiplier, INaL_Multiplier,
        IKr_Multiplier, IKs_Multiplier, IK1_Multiplier, IKb_Multiplier,
        INaCa_Multiplier, INaK_Multiplier, INab_Multiplier, ICab_Multiplier,
        IpCa_Multiplier, ICaCl_Multiplier, IClb_Multiplier,
        Jrel_Multiplier, Jup_Multiplier,
        nao, cao, ko, ICaL_fractionSS, INaCa_fractionSS,
        stimAmp, stimDur, None, None, None
    )

    nPoints = sum(len(time[i]) for i in range(ignore_first_spikes, len(time)))

    keys = ['INa','INaL','Ito','ICaL','IKr','IKs','IK1','INaCa_i','INaCa_ss',
            'INaK','IKb','INab','ICab','IpCa','Jdiff','JdiffNa','JdiffK',
            'Jup','Jleak','Jtr','Jrel','CaMKa','Istim',
            'fINap','fINaLp','fICaLp','fJrelp','fJupp',
            'CaJSR','CaNSR','PhiCaL_ss','V','Ta','lambda']

    currents = {k: np.zeros(nPoints) for k in keys}
    currents['time'] = np.zeros(nPoints)
    currents['Cai']  = np.zeros(nPoints)
    currents['Cass'] = np.zeros(nPoints)
    currents['Ta'] = np.zeros(nPoints)
    currents['lambda'] = np.zeros(nPoints)

    idx = 0
    for iBeat, (t_beat, X_beat) in enumerate(zip(time, X)):
        if iBeat < ignore_first_spikes:
            continue
        for j in range(len(t_beat)):
            t_j = t_beat[j]
            xj  = X_beat[j]
            IsJs = model_fn(t_j, xj, 0, *extra_args)
            currents['time'][idx] = iBeat * bcl + t_j
            currents['Cai'][idx]  = xj[5]
            currents['Cass'][idx] = xj[6]
            for ki_, key in enumerate(keys):
                currents[key][idx] = IsJs[ki_]
            idx += 1

    currents['INaCa'] = currents['INaCa_i'] + currents['INaCa_ss']
    return currents


# ---------------------------------------------------------------------------
# getStartingState
# ---------------------------------------------------------------------------

def get_starting_state(cell_type: str = 'm_endo') -> np.ndarray:
    """
    Return a steady-state initial condition vector (length 49).

    cell_type : 'm_endo' | 'm_epi' | 'm_mid'
    """
    states = {
        'm_endo': [-88.6369922306458, 11.8973412949238, 11.8976610470850,
                   141.234464714982, 141.234423402713, 7.26747296460659e-05,
                   6.33786975780735e-05, 1.53265306371970, 1.53394579180493,
                   0.000828007761976018, 0.666527193684116, 0.826020806005678,
                   0.826055985895856, 0.825850881115628, 0.000166868626513013,
                   0.522830604669169, 0.285969584294187, 0.000959137028030184,
                   0.999601150012565, 0.593401639836100, 0.000488696137242056,
                   0.999601147267179, 0.654668660159696, 9.50007519781516e-32,
                   0.999999992317577, 0.939258048397962, 0.999999992317557,
                   0.999898379647465, 0.999978251560040, 0.000444816183420527,
                   0.000755072490632667, 0.999999992318446, 0.999999992318445,
                   0.242404683449520, 0.000179537726989804, -6.88308558109975e-25,
                   0.0111749845355653, 0.998036620213316, 0.000858801779013532,
                   0.000709744678350176, 0.000381261722195702, 1.35711566929992e-05,
                   2.30252452954649e-23, 0.000156194131630688, 0.000235128869057516,
                   0.00807763107171867, 0.999373435811656, 0, 0],

        'm_epi':  [-89.0462806262884, 12.7218980311997, 12.7222039977392,
                   142.248960281735, 142.248911688304, 6.54105789316085e-05,
                   5.68443136844764e-05, 1.80911728399381, 1.80970235621251,
                   0.000758182108180449, 0.679839847935577, 0.834150231581688,
                   0.834188252920967, 0.834081731522592, 0.000154387698861246,
                   0.538295069820379, 0.302769394159465, 0.000933035060391086,
                   0.999628705730844, 0.999626204093615, 0.000475390662092180,
                   0.999628705544664, 0.999628513430851, 1.74213411952898e-37,
                   0.999999993122906, 0.947952168523141, 0.999999993122889,
                   0.999932686646139, 0.999982915381882, 0.000291544679470133,
                   0.000502604507932921, 0.999999993124187, 0.999999993123756,
                   0.228815500940270, 0.000171497784228012, -1.13118992668881e-26,
                   0.0129505221481656, 0.998194356754674, 0.000834232097912889,
                   0.000683865770895308, 0.000277878501096440, 9.66775862738005e-06,
                   8.16930403133409e-24, 0.000125999575445634, 0.000189952183128226,
                   0.00655149435193622, 0.999493972060333, 0, 0],

        'm_mid':  [-89.5379994049964, 14.9292004720038, 14.9296673679334,
                   144.844718688810, 144.844658476157, 7.50228807455408e-05,
                   6.10763598140135e-05, 1.79043480744558, 1.79484249993962,
                   0.000681936485046493, 0.695380653101535, 0.843488797335149,
                   0.843520761455969, 0.843226224403045, 0.000140621109700401,
                   0.545314876174586, 0.292496735833565, 0.000902612655601118,
                   0.999659345906191, 0.563119679366890, 0.000459883274920751,
                   0.999659343029625, 0.623696443871387, -1.31418873360667e-33,
                   0.999999993979673, 0.920408593154793, 0.999999993979652,
                   0.999761950174748, 0.999962530196306, 0.000385359469667100,
                   0.000853529194511867, 0.999999993978835, 0.999999993980401,
                   0.266415111925392, 0.000162310655612839, 1.20976169203982e-24,
                   0.0178243652102213, 0.997971986641796, 0.000805399061926759,
                   0.000678179976274546, 0.000526536308931670, 1.78956481798154e-05,
                   7.05916237956270e-23, 0.000167065448972034, 0.000250679417924283,
                   0.00860262481216925, 0.999331445205816, 0, 0],
    }
    if cell_type not in states:
        raise ValueError(f'Unknown cell_type "{cell_type}". Choose from: {list(states.keys())}')
    return np.array(states[cell_type])


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nbeats = 100
    print('Running ', nbeats, ' beats of m_endo at BCL=1000 ms ...')
    X0_endo = get_starting_state('m_endo')
    params = {'bcl': 1000, 'model': model_ToRORd_Land}
    t_out_endo, X_out, params = model_runner(X0_endo, params, beats=nbeats, ignore_first=4)
    X_endo = get_currents_structure(t_out_endo, X_out, params)
    print('Running ', nbeats, ' beats of m_epi at BCL=1000 ms ...')
    X0_epi = get_starting_state('m_epi')
    params = {'bcl': 1000, 'model': model_ToRORd_Land}
    t_out_epi, X_out, params = model_runner(X0_epi, params, beats=nbeats, ignore_first=4)
    X_epi = get_currents_structure(t_out_epi, X_out, params)
    print('Running ', nbeats, ' beats of m_mid at BCL=1000 ms ...')
    X0_mid = get_starting_state('m_mid')
    params = {'bcl': 1000, 'model': model_ToRORd_Land}
    t_out_mid, X_out, params = model_runner(X0_mid, params, beats=nbeats, ignore_first=4)
    X_mid = get_currents_structure(t_out_mid, X_out, params)

    # flatten the single retained beat
    t_endo = t_out_endo[0]
    t_epi = t_out_epi[0]
    t_mid = t_out_mid[0]
    V_endo = X_endo['V']
    Cai_endo = X_endo['Cai']
    Ta_endo = X_endo['Ta']
    V_epi = X_epi['V']
    Cai_epi = X_epi['Cai']
    Ta_epi = X_epi['Ta']
    V_mid = X_mid['V']
    Cai_mid = X_mid['Cai']
    Ta_mid = X_mid['Ta']

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_endo, V_endo, t_epi, V_epi, t_mid, V_mid)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.subplot(3, 1, 2)
    plt.plot(t_endo, Cai_endo, t_epi, Cai_epi, t_mid, Cai_mid)
    plt.xlabel('Time (ms)')
    plt.ylabel('Calcium transient (mM)')
    plt.subplot(3, 1, 3)
    plt.plot(t_endo, Ta_endo, t_epi, Ta_epi, t_mid, Ta_mid)
    plt.xlabel('Time (ms)')
    plt.ylabel('Active Tension (kPa)')
    plt.title('ToRORd-Land')
    plt.legend(['Endo', 'Epi', 'Mid'])
    plt.tight_layout()
    plt.savefig('ap_demo.png', dpi=150)
    print('Saved ap_demo.png')
    plt.show()
