import numpy as np
import itertools as it
from matplotlib import pyplot as plt
import numba
from numba import njit
import datetime
import logging
import time

# Administrative setup
logging.basicConfig(
    filename="runs.log", level=logging.INFO, format=logging.BASIC_FORMAT, filemode="w"
)
logger = logging.Logger(__name__, level=logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(filename="runs.log", mode="w"))


# Read inputs

if False:
    start_sim = int(input("What to do? "))

    # Only support new simulations initially
    if start_sim != 1:
        exit()

    Nx = int(input("N="))
    My = int(input("M="))

    use_jet = input("jet?")
    # no jets right now
    if use_jet != "n":
        exit()

    Re = np.float(input("Reynolds"))
    a_tilde = np.float(input("a tilde"))

    timesteps = int(input("timesteps"))
    report_timesteps = int(input("report timesteps"))

    dt = np.float(input("dt"))
    tolerance_level = np.float(input("tolerance"))
    BCs = np.int(input("BCs"))

    output_file_name = input("output name")
    iterations_between_writes = input("iterations between writes")

    if input("continue") != "y":
        exit()
else:
    Nx = 199
    My = 399
    Re = 1600
    timesteps = 250_000
    report_timesteps = 50
    a_tilde = 1.5
    dt = 0.00035
    tolerance_level = 1e-5
    BCs = 20
    output_file_name = "testout"
    iterations_between_writes = -1
    enable_jet = True
    ia = 120
    ib = 130
    c0 = 0.01
    freq = 0.06


# Set up initial conditions
Xmin = -20
Xmax = 20
Ymin = 1
Ymax = 11

x = np.linspace(Xmin, Xmax, Nx + 2, dtype=np.double)
y = np.linspace(Ymin, Ymax, My + 2, dtype=np.double)

dx = (Xmax - Xmin) / (Nx + 1)
dy = (Ymax - Ymin) / (My + 1)
dxx = dx ** 2
dyy = dy ** 2
dx2 = 2 * dx
dy2 = 2 * dy

yv, xv = np.meshgrid(y, x)
DM = np.sqrt(np.square(xv) + np.square(yv))
DM2 = np.square(DM)

# Initialize key simulation variables
V = -(yv - 1) / DM
U = (xv + a_tilde) / DM

Psi = (xv + a_tilde) * (yv - 1)

Omega = np.zeros_like(Psi, dtype=np.double)
Omega[:, 0] = (7 * Psi[:, 0] - 8 * Psi[:, 1] + Psi[:, 2]) / (2 * dyy * DM2[:, 0])

# Calculation helper variables
Kappa2 = (dx / dy) ** 2
KappaA = 1 / (2 * (1 + Kappa2))
Rc = Re * dx

Cx = dt / dx
Cx2 = 0.5 * Cx
Cy = dt / dy
Cy2 = 0.5 * Cy

alphaX = dt / (dxx * Re)
alphaY = dt / (dyy * Re)
alpha = 2.0 * alphaX + 2.0 * alphaY

# Calculation functions
@njit(parallel=True)
def omega_calc(Nx, My, Cx2, Cy2, alpha, alphaX, alphaY, Omega, Omega0, U, V, DM, DM2):
    for i in numba.prange(1, Omega.shape[0] - 1):
        for j in numba.prange(1, Omega.shape[1] - 1):
            Omega[i, j] = (
                Omega0[i, j] * (1 - alpha / DM2[i, j])
                + (
                    Omega0[i + 1, j]
                    * (-Cx2 * U[i + 1, j] * DM[i + 1, j] + alphaX)
                    / DM2[i, j]
                )
                + (
                    Omega0[i - 1, j]
                    * (Cx2 * U[i - 1, j] * DM[i - 1, j] + alphaX)
                    / DM2[i - 1, j]
                )
                + (
                    Omega0[i, j + 1]
                    * (-Cy2 * V[i, j + 1] * DM[i, j + 1] + alphaY)
                    / DM2[i, j]
                )
                + (
                    Omega0[i, j - 1]
                    * (Cy2 * V[i, j - 1] * DM[i, j - 1] + alphaY)
                    / DM2[i, j]
                )
            )


@njit(parallel=True)
def psi_calc(Nx, My, Kappa2, KappaA, dxx, Psi, Omega, DM2, tolerance_level):
    psitol = 1
    kpsi = 0
    while psitol > tolerance_level:
        kpsi += 1
        Psi0 = (
            Psi.copy()
        )  # Copying in psi actually increases the MSE compared to the fortran solver?
        for i in numba.prange(1, Psi.shape[0] - 1):
            for j in numba.prange(1, Psi.shape[1] - 1):
                Psi[i, j] = KappaA * (
                    dxx * Omega[i, j] * DM2[i, j]
                    + Psi0[i + 1, j]
                    + Psi0[i - 1, j]
                    + Kappa2 * (Psi0[i, j + 1] + Psi0[i, j - 1])
                )
        psitol = np.max(np.abs(Psi - Psi0))
    return kpsi


@njit()
def u_calc(Nx, My, i, j, dy2, Psi, U, DM):
    U[i, j] = (Psi[i, j + 1] - Psi[i, j - 1]) / (dy2) / DM[i, j]


@njit()
def v_calc(Nx, My, i, j, dx2, Psi, V, DM):
    V[i, j] = -(Psi[i + 1, j] - Psi[i - 1, j]) / (dx2) / DM[i, j]


@njit(parallel=True)
def side_boundaries(Nx, My, a_tilde, DM, U, V, Psi, Omega):
    i = 0
    for j in numba.prange(1, My + 1):
        if j > BCs:
            Omega[i, j] = 0
            Psi[i, j] = (x[i] + a_tilde) * (y[j] - 1)
            U[i, j] = (x[i] + a_tilde) / DM[i, j]
            V[i, j] = -(y[j] - 1) / DM[i, j]
        else:
            Omega[i, j] = Omega[i + 1, j]
            Psi[i, j] = Psi[i + 1, j]
            U[i, j] = U[i + 1, j]
            V[i, j] = V[i + 1, j]

    i = Nx + 1
    for j in numba.prange(1, My + 2):
        if j > BCs:
            Omega[i, j] = 0
            Psi[i, j] = (x[i] + a_tilde) * (y[j] - 1)
            U[i, j] = (x[i] + a_tilde) / DM[i, j]
            V[i, j] = -(y[j] - 1) / DM[i, j]
        else:
            Omega[i, j] = Omega[i - 1, j]
            Psi[i, j] = Psi[i - 1, j]
            U[i, j] = U[i - 1, j]
            V[i, j] = V[i - 1, j]


@njit(parallel=True)
def velocity_calculations(Nx, My, dx2, dy2, U, V, DM, Psi):
    for i in numba.prange(1, Nx + 1):
        for j in numba.prange(1, My + 1):
            U[i, j] = (Psi[i, j + 1] - Psi[i, j - 1]) / (dy2) / DM[i, j]
            V[i, j] = -(Psi[i + 1, j] - Psi[i - 1, j]) / (dx2) / DM[i, j]


@njit(parallel=True)
def no_jet_upper_lower_boundaries(Nx, My, Psi, Omega, U, V, dyy, DM):
    # Lower
    d6 = np.square(x) + np.square(y[0])

    Psi[:, 0] = 0
    Omega[:, 0] = (7.0 * Psi[:, 0] - 8.0 * Psi[:, 1] + Psi[:, 2]) / (2.0 * dyy) / d6
    U[:, 0] = 0
    V[:, 0] = 0

    # Upper
    Omega[:, -1] = 0
    Psi[:, -1] = (x + a_tilde) * (y[-1] - 1)
    U[:, -1] = (x + a_tilde) / DM[:, -1]
    V[:, -1] = -(y[-1] - 1) / DM[:, -1]


@njit()
def jet_upper_lower_boundaries(
    Nx, My, Psi, Omega, U, V, dyy, DM, ia, ib, c0, freq, t, xv, yv
):
    amewa = (ia - ((Nx + 1) / 2 + 1)) * dx
    f = np.sin(2 * np.pi * freq * t)

    # Lower
    d6 = np.square(x) + np.square(y[0])

    Psi[:ia, 0] = 0

    Psi[ia : ib + 1, 0] = (
        -c0
        * (
            0.0
            - 0.5 * amewa * np.sqrt(amewa ** 2 + 1)
            - 0.5 * np.sinh(amewa)
            + 0.5 * xv[ia : ib + 1, 0] * np.sqrt(xv[ia : ib + 1, 0] ** 2 + 1)
            + 0.5 * np.sinh(xv[ia : ib + 1, 0])
        )
    ) * f

    Psi[ib + 1 :, 0] = Psi[ib, 0]

    Omega[:, 0] = (7.0 * Psi[:, 0] - 8.0 * Psi[:, 1] + Psi[:, 2]) / (2.0 * dyy) / d6
    U[:, 0] = 0
    V[:, 0] = 0
    V[ia + 1 : ib] = c0 * f

    # Upper
    Omega[:, -1] = 0
    Psi[:, -1] = (x + a_tilde) * (y[-1] - 1)
    U[:, -1] = (x + a_tilde) / DM[:, -1]
    V[:, -1] = -(y[-1] - 1) / DM[:, -1]


@njit()
def run_iteration(
    Nx,
    My,
    Cx2,
    Cy2,
    alpha,
    alphaX,
    alphaY,
    Omega,
    Omega0,
    U,
    V,
    DM,
    DM2,
    Kappa2,
    KappaA,
    dxx,
    dyy,
    Psi,
    tolerance_level,
    iteration,
    xv,
    yv,
):
    t = iteration * dt

    omega_calc(Nx, My, Cx2, Cy2, alpha, alphaX, alphaY, Omega, Omega0, U, V, DM, DM2)
    kpsi = psi_calc(Nx, My, Kappa2, KappaA, dxx, Psi, Omega, DM2, tolerance_level)

    # Boundary conditions

    if enable_jet:
        jet_upper_lower_boundaries(
            Nx, My, Psi, Omega, U, V, dyy, DM, ia, ib, c0, freq, t, xv, yv
        )
    else:
        no_jet_upper_lower_boundaries(Nx, My, Psi, Omega, U, V, dyy, DM)

    # Sides
    side_boundaries(Nx, My, a_tilde, DM, U, V, Psi, Omega)

    # Velocity calculations
    velocity_calculations(Nx, My, dx2, dy2, U, V, DM, Psi)

    return kpsi


@njit()
def total_velocity(U, V):
    return np.sqrt(np.square(U) + np.square(V))


def state_to_image(xv, yv, U, V, iteration):
    real_x = 0.5 * (np.square(xv) - np.square(yv))
    real_y = xv * yv
    vel = total_velocity(U, V)
    fig, ax = plt.subplots()
    cs = ax.contourf(real_x, real_y, vel, 300)
    ax.set_xlim([-2, 16])
    ax.set_ylim([-4, 12])
    fig.colorbar(cs, ax=ax)
    plt.savefig(f"VizOut/flow{iteration:07}.png")
    plt.close()
    del fig


# Loop and continually calculate U,V,Omega,Psi
Omega0 = Omega.copy()
start_time = time.time()
for iteration in range(timesteps):
    Omega0 = Omega.copy()
    Psi0 = Psi.copy()
    kpsi = run_iteration(
        Nx,
        My,
        Cx2,
        Cy2,
        alpha,
        alphaX,
        alphaY,
        Omega,
        Omega0,
        U,
        V,
        DM,
        DM2,
        Kappa2,
        KappaA,
        dxx,
        dyy,
        Psi,
        tolerance_level,
        iteration,
        xv,
        yv,
    )
    omtol = np.max(np.abs(Omega - Omega0))
    psitol = np.max(np.abs(Psi - Psi0))

    if not iteration % 3000:
        current_time = time.time()
        iteration_rate = (current_time - start_time) / (iteration + 1)
        logger.info(
            f"{iteration:04} {kpsi:04} {omtol:e} {psitol:e} {iteration:04} {datetime.datetime.today()} {(current_time-start_time):.2f}s {iteration_rate:.4f}s/it {(timesteps-iteration)*iteration_rate/60:.0f}m to completion"
        )
        state_to_image(xv, yv, U, V, iteration)
