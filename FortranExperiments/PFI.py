from os import times
import numpy as np
import itertools as it
from matplotlib import pyplot as plt
import numba
from numba import njit

with open("testout", "r") as f:
    data = f.readlines()
    odata = data[:401]
    pdata = data[401 : 401 * 2]
    udata = data[401 * 2 : 401 * 3]
    vdata = data[401 * 3 : 401 * 4]


testout = np.fromstring("".join(odata), dtype=np.float, sep=" ")
testout = np.reshape(testout, (401, 201)).T

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
    Re = 700
    timesteps = 10000
    report_timesteps = 50
    a_tilde = 0
    dt = 1e-4
    tolerance_level = 1e-5
    BCs = 20
    output_file_name = "testout"
    iterations_between_writes = -1


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

U_history = np.zeros((*U.shape, timesteps // report_timesteps))
V_history = np.zeros((*V.shape, timesteps // report_timesteps))

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
    while psitol > tolerance_level:
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
def velocity_calculations(Nx, My, dx2, dy2, U, V, DM):
    for i in numba.prange(1, Nx + 1):
        for j in numba.prange(1, My + 1):
            u_calc(Nx, My, i, j, dy2, Psi, U, DM)
            v_calc(Nx, My, i, j, dx2, Psi, V, DM)


@njit(parallel=True)
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
):
    Omega0 = Omega.copy()

    omega_calc(Nx, My, Cx2, Cy2, alpha, alphaX, alphaY, Omega, Omega0, U, V, DM, DM2)
    psi_calc(Nx, My, Kappa2, KappaA, dxx, Psi, Omega, DM2, tolerance_level)

    # Boundary conditions
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

    # Sides
    side_boundaries(Nx, My, a_tilde, DM, U, V, Psi, Omega)

    # Velocity calculations
    velocity_calculations(Nx, My, dx2, dy2, U, V, DM)

    # Omtol
    # print(np.max(np.abs(Omega - Omega0)))


# Loop and continually calculate U,V,Omega,Psi
Omega0 = Omega.copy()
for iteration in range(timesteps):
    run_iteration(
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
    )

print(np.mean(np.square(Omega - testout)))


# plotting logic
if False:
    real_x = 0.5 * (np.square(xv) - np.square(yv))
    real_y = xv * yv
    total_vel = np.sqrt(np.square(U) + np.square(V))

    for iteration in range(0, timesteps, timesteps // report_timesteps):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.contourf(real_x, real_y, total_vel, np.linspace(0, 3, 100))
        ax2.contourf(total_vel, np.linspace(0, 3, 100))
        plt.savefig(f"VizOut/flow{iteration:04}.png")
        plt.close()
        del fig, ax1, ax2
