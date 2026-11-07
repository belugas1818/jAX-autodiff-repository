"""Black-Scholes prices and Greeks, written without using the Monte Carlo pricer.

The core idea: the log of a GBM spot is normal, so the expectation of a
call or a put collapses to two normal cdfs.
"""

from __future__ import annotations

import math


def norm_pdf(x: float) -> float:
    """Standard normal density.

    The density is the derivative of the cdf. Its formula is

        n(x) = (1 / sqrt(2 * pi)) * exp(-0.5 * x * x)

    because the Gaussian integral of exp(-0.5 * x * x) from -infinity to
    infinity equals sqrt(2 * pi), so dividing by that constant makes the
    density integrate to 1.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cdf N(x) = P(Z <= x) for Z ~ N(0, 1).

    The error function is defined by

        erf(y) = (2 / sqrt(pi)) * integral_0^y exp(-t * t) dt

    and it is an odd function. Change variables in the Gaussian integral.
    Let t = z / sqrt(2), so dz = sqrt(2) * dt and

        N(x) = integral_{-infinity}^{x} (1 / sqrt(2 * pi)) * exp(-0.5 * z * z) dz
             = 0.5 + integral_0^{x} (1 / sqrt(2 * pi)) * exp(-0.5 * z * z) dz.

    The substitution z = sqrt(2) * t gives exp(-0.5 * z * z) = exp(-t * t) and

        integral_0^{x} exp(-0.5 * z * z) dz
            = sqrt(2) * integral_0^{x / sqrt(2)} exp(-t * t) dt
            = sqrt(2) * (sqrt(pi) / 2) * erf(x / sqrt(2))
            = sqrt(2 * pi) / 2 * erf(x / sqrt(2)).

    Divide by sqrt(2 * pi):

        integral_0^{x} n(z) dz = 0.5 * erf(x / sqrt(2)).

    Therefore

        N(x) = 0.5 + 0.5 * erf(x / sqrt(2))
             = 0.5 * (1 + erf(x / sqrt(2))).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(s0: float, strike: float, r: float, q: float, sigma: float, maturity: float):
    """Compute the two Black-Scholes distances d1 and d2.

    Under the risk-neutral measure,

        log(S_T) ~ Normal(log(S_0) + (r - q - 0.5 * sigma * sigma) * T,
                          sigma * sqrt(T)).

    Divide the distance from log(K) to that mean by the standard deviation:

        d2 = (log(S_0 / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

    which is exactly (E[log S_T] - log K) / std(log S_T). Then

        d1 = d2 + sigma * sqrt(T)
           = (log(S_0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T)).

    d1 is the same standardised distance after the stock-numeraire measure
    change, whose drift is r - q + 0.5 * sigma * sigma instead of
    r - q - 0.5 * sigma * sigma. The two measures differ by one volatility
    unit sigma * sqrt(T), which is why d1 - d2 = sigma * sqrt(T).
    """
    vol_sqrt_t = sigma * math.sqrt(maturity)
    d2 = (
        math.log(s0 / strike) + (r - q - 0.5 * sigma * sigma) * maturity
    ) / vol_sqrt_t
    d1 = d2 + vol_sqrt_t
    return d1, d2


def bs_price(
    s0: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    payoff: str,
) -> float:
    """Closed-form Black-Scholes price of a call, a put, or a cash digital call.

    Write df_r = exp(-r * T) and df_q = exp(-q * T). S_T has the lognormal
    law above, and N is ``norm_cdf``.

    Call. The payoff is max(S_T - K, 0) = (S_T - K) * 1_{S_T >= K}.

        exp(-r * T) * E[(S_T - K) * 1_{S_T >= K}]
            = exp(-r * T) * E[S_T * 1_{S_T >= K}] - K * exp(-r * T) * P(S_T >= K).

    P(S_T >= K) = P(log S_T >= log K) = N(d2), because (log S_T - mean) / std
    exceeds -d2 with probability N(d2). For the stock term, change to the
    stock measure (multiply the density by the growth of the reinvested
    stock). Under that measure the probability of finishing in the money is
    N(d1), and

        exp(-r * T) * E[S_T * 1_{S_T >= K}] = S_0 * exp(-q * T) * N(d1).

    Hence

        call = S_0 * df_q * N(d1) - K * df_r * N(d2).

    Put. Put-call parity for the prepaid forward is

        call - put = S_0 * df_q - K * df_r

    because the portfolio long call, short put pays S_T - K for every path,
    whose present value is the prepaid forward minus the prepaid cash strike.
    Solving for the put,

        put = call - S_0 * df_q + K * df_r
            = K * df_r * N(-d2) - S_0 * df_q * N(-d1)

    after using N(x) + N(-x) = 1 on each term.

    Cash-or-nothing digital call. The contract pays 1 cash if S_T >= K and 0
    otherwise, so there is no S_T term:

        digital_call = exp(-r * T) * P(S_T >= K) = df_r * N(d2).
    """
    d1, d2 = _d1_d2(s0, strike, r, q, sigma, maturity)
    df_r = math.exp(-r * maturity)
    df_q = math.exp(-q * maturity)
    if payoff == "call":
        return s0 * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2)
    if payoff == "put":
        return strike * df_r * norm_cdf(-d2) - s0 * df_q * norm_cdf(-d1)
    if payoff == "digital_call":
        return df_r * norm_cdf(d2)
    raise ValueError(f"no closed form for payoff {payoff}")


def bs_greeks(
    s0: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    payoff: str,
) -> dict:
    """Closed-form Greeks. Theta is the calendar decay -dV/dT.

    Notation: df_q = exp(-q * T), df_r = exp(-r * T), n = ``norm_pdf``,
    N = ``norm_cdf``, v = sigma * sqrt(T). S, K, T stand for s0, strike,
    maturity. Every Greek below is the partial derivative of ``bs_price``
    with the other inputs held fixed.

    Vanilla call and put share gamma and vega. Differentiating the call

        C = S * df_q * N(d1) - K * df_r * N(d2)

    with respect to S, the density terms cancel by the identity
    S * df_q * n(d1) = K * df_r * n(d2), and what remains is

        delta_call = df_q * N(d1).

    Put-call parity C - P = S * df_q - K * df_r has S-derivative df_q, so

        delta_put = delta_call - df_q = -df_q * N(-d1).

    Differentiating delta_call once more,

        d(d1)/dS = 1 / (S * v)

        gamma = df_q * n(d1) * d(d1)/dS = df_q * n(d1) / (S * v).

    The put has the same gamma because parity's second S-derivative is 0.

    Vega is dV/d(sigma), not dV per volatility point. The same cancellation
    of the two density terms leaves

        vega = S * df_q * n(d1) * sqrt(T)

    for both the call and the put.

    Rho is dV/dr. From the call formula the strike discount contributes
    K * T * df_r * N(d2), and the d1, d2 derivatives cancel, so

        rho_call = K * T * df_r * N(d2).

    Parity has dr derivative K * T * df_r, therefore

        rho_put = rho_call - K * T * df_r = -K * T * df_r * N(-d2).

    Theta is -dV/dT (a year passes, expiry moves one year closer).
    Differentiating the call and using S * df_q * n(d1) = K * df_r * n(d2)
    collapses the d1 and d2 derivatives into one density term:

        dC/dT = -q * S * df_q * N(d1) + r * K * df_r * N(d2)
                + S * df_q * n(d1) * sigma / (2 * sqrt(T))

        theta_call = -dC/dT
                    = -S * df_q * n(d1) * sigma / (2 * sqrt(T))
                      - r * K * df_r * N(d2)
                      + q * S * df_q * N(d1).

    Parity: dC/dT - dP/dT = -q * S * df_q + r * K * df_r, so
    theta_call - theta_put = q * S * df_q - r * K * df_r, and

        theta_put = -S * df_q * n(d1) * sigma / (2 * sqrt(T))
                    + r * K * df_r * N(-d2)
                    - q * S * df_q * N(-d1).

    Cash digital call: DC = df_r * N(d2), a pure probability, so the
    derivatives hit the discount and d2 only.

        d(d2)/dS = 1 / (S * v)
        delta = df_r * n(d2) / (S * v)

        d(delta)/dS = -df_r * n(d2) * d1 / (S * S * sigma * sigma * T)
        because d(n(d2))/dS = -d2 * n(d2) / (S * v) and d1 = d2 + v.
        That quotient is gamma.

        d(d2)/d(sigma) = -d1 / sigma
        vega = df_r * n(d2) * (-d1 / sigma)

        d(d2)/dr = sqrt(T) / sigma
        rho = -T * df_r * N(d2) + df_r * n(d2) * sqrt(T) / sigma

        d2 = num / den with
            num = log(S / K) + (r - q - 0.5 * sigma * sigma) * T
            den = sigma * sqrt(T)
        d(d2)/dT = (r - q - 0.5 * sigma * sigma) / den - d2 / (2 * T)
        d(DC)/dT = -r * df_r * N(d2) + df_r * n(d2) * d(d2)/dT
        theta = -d(DC)/dT.
    """
    d1, d2 = _d1_d2(s0, strike, r, q, sigma, maturity)
    df_r = math.exp(-r * maturity)
    df_q = math.exp(-q * maturity)
    sqrt_t = math.sqrt(maturity)
    pdf_d1 = norm_pdf(d1)
    price = bs_price(s0, strike, r, q, sigma, maturity, payoff)

    if payoff in ("call", "put"):
        gamma = df_q * pdf_d1 / (s0 * sigma * sqrt_t)
        vega = s0 * df_q * pdf_d1 * sqrt_t
        if payoff == "call":
            delta = df_q * norm_cdf(d1)
            rho = strike * maturity * df_r * norm_cdf(d2)
            theta = (
                -s0 * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)
                - r * strike * df_r * norm_cdf(d2)
                + q * s0 * df_q * norm_cdf(d1)
            )
        else:
            delta = -df_q * norm_cdf(-d1)
            rho = -strike * maturity * df_r * norm_cdf(-d2)
            theta = (
                -s0 * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)
                + r * strike * df_r * norm_cdf(-d2)
                - q * s0 * df_q * norm_cdf(-d1)
            )
    elif payoff == "digital_call":
        pdf_d2 = norm_pdf(d2)
        delta = df_r * pdf_d2 / (s0 * sigma * sqrt_t)
        gamma = -df_r * pdf_d2 * d1 / (s0 * s0 * sigma * sigma * maturity)
        vega = -df_r * pdf_d2 * d1 / sigma
        rho = df_r * (-maturity * norm_cdf(d2) + pdf_d2 * sqrt_t / sigma)
        d2_dT = (r - q - 0.5 * sigma * sigma) / (sigma * sqrt_t) - d2 / (2.0 * maturity)
        theta = -(-r * df_r * norm_cdf(d2) + df_r * pdf_d2 * d2_dT)
    else:
        raise ValueError(f"no closed form for payoff {payoff}")

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta,
    }
