/*
 * Parallel Cardiac Electrophysiology Simulation
 * Fenton-Karma 3-Variable Ionic Model on 2D Tissue
 * 
 * Decomposition Strategies:
 *   1 = Row-block
 *   2 = Column-block  
 *   3 = 2D-block (checkerboard)
 *
 * Usage: mpirun -np <P> ./cardiac_sim <strategy> <grid_size> <num_steps> <snapshot_interval>
 * Example: mpirun -np 8 ./cardiac_sim 3 2048 5000 2500
 *
 * Course: Parallel & Distributed Computing (CS-315) — NUST SINES
 * Reference: Fenton & Karma, Chaos 8(1), 1998
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Fenton-Karma Model Parameters (Parameter Set 1 - Human Ventricle)
 * Reference: Fenton & Karma, Chaos 8(1), 1998
 * ============================================================ */
#define TAU_V_PLUS   3.33
#define TAU_V1_MINUS 19.6
#define TAU_V2_MINUS 1000.0
#define TAU_W_PLUS   667.0
#define TAU_W_MINUS  11.0
#define TAU_D        0.25
#define TAU_0        8.3
#define TAU_R        50.0
#define TAU_SI       45.0
#define U_C          0.13
#define U_V          0.04
#define U_C_SI       0.85
#define K_FK         10.0

/* Tissue parameters */
#define DX    0.025    /* spatial step (cm) */
#define DY    0.025
#define DT    0.1      /* time step (ms) */
#define DIFF  0.001    /* diffusion coefficient (cm^2/ms) */

/* ============================================================
 * Fenton-Karma ionic currents
 * ============================================================ */
static inline double heaviside(double x) {
    return (x >= 0.0) ? 1.0 : 0.0;
}

static void fk_step(double u, double v, double w,
                     double *du_ion, double *dv, double *dw)
{
    double p = heaviside(u - U_C);
    double q = heaviside(u - U_V);

    /* Fast inward current (Na+) */
    double J_fi = -v * p * (1.0 - u) * (u - U_C) / TAU_D;

    /* Slow outward current (K+) */
    double J_so = u * (1.0 - p) / TAU_0 + p / TAU_R;

    /* Slow inward current (Ca2+) */
    double J_si = -w * (1.0 + tanh(K_FK * (u - U_C_SI))) / (2.0 * TAU_SI);

    *du_ion = -(J_fi + J_so + J_si);

    /* Gate variable v (fast recovery) */
    double tau_v_minus = q * TAU_V1_MINUS + (1.0 - q) * TAU_V2_MINUS;
    *dv = (1.0 - p) * (1.0 - v) / tau_v_minus - p * v / TAU_V_PLUS;

    /* Gate variable w (slow recovery) */
    *dw = (1.0 - p) * (1.0 - w) / TAU_W_MINUS - p * w / TAU_W_PLUS;
}

/* ============================================================
 * Decomposition
 * ============================================================ */
typedef struct {
    int strategy;
    int N;
    int local_rows, local_cols;
    int row_start, col_start;
    int rank, nprocs;
    int north, south, east, west;
    int proc_rows, proc_cols;
    MPI_Comm cart_comm;
    /* Pre-allocated column halo buffers (avoids per-step malloc/free) */
    double *col_send;
    double *col_recv;
} Decomp;

static void setup_decomposition(Decomp *d, int strategy, int N, int rank, int nprocs) {
    d->strategy = strategy;
    d->N = N;
    d->rank = rank;
    d->nprocs = nprocs;
    d->cart_comm = MPI_COMM_WORLD;
    d->north = d->south = d->east = d->west = -1;

    if (strategy == 1) {
        /* Row-block */
        int base = N / nprocs, extra = N % nprocs;
        d->local_rows = base + (rank < extra ? 1 : 0);
        d->local_cols = N;
        d->row_start = rank * base + (rank < extra ? rank : extra);
        d->col_start = 0;
        d->north = (rank > 0) ? rank - 1 : -1;
        d->south = (rank < nprocs - 1) ? rank + 1 : -1;
        d->proc_rows = nprocs;
        d->proc_cols = 1;
    }
    else if (strategy == 2) {
        /* Column-block */
        int base = N / nprocs, extra = N % nprocs;
        d->local_rows = N;
        d->local_cols = base + (rank < extra ? 1 : 0);
        d->row_start = 0;
        d->col_start = rank * base + (rank < extra ? rank : extra);
        d->east = (rank < nprocs - 1) ? rank + 1 : -1;
        d->west = (rank > 0) ? rank - 1 : -1;
        d->proc_rows = 1;
        d->proc_cols = nprocs;
    }
    else {
        /* 2D-block with MPI Cartesian topology */
        int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
        MPI_Dims_create(nprocs, 2, dims);
        d->proc_rows = dims[0];
        d->proc_cols = dims[1];
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &d->cart_comm);

        /* After Cart_create with reorder=1, rank may differ from COMM_WORLD */
        MPI_Comm_rank(d->cart_comm, &d->rank);
        MPI_Cart_coords(d->cart_comm, d->rank, 2, coords);

        int pr = coords[0], pc = coords[1];
        int base_r = N / dims[0], extra_r = N % dims[0];
        d->local_rows = base_r + (pr < extra_r ? 1 : 0);
        d->row_start = pr * base_r + (pr < extra_r ? pr : extra_r);

        int base_c = N / dims[1], extra_c = N % dims[1];
        d->local_cols = base_c + (pc < extra_c ? 1 : 0);
        d->col_start = pc * base_c + (pc < extra_c ? pc : extra_c);

        MPI_Cart_shift(d->cart_comm, 0, 1, &d->north, &d->south);
        MPI_Cart_shift(d->cart_comm, 1, 1, &d->west, &d->east);
    }

    /* Pre-allocate column halo buffers (used every timestep in exchange_halos) */
    if (d->west >= 0 || d->east >= 0) {
        d->col_send = (double *)malloc(d->local_rows * sizeof(double));
        d->col_recv = (double *)malloc(d->local_rows * sizeof(double));
    } else {
        d->col_send = NULL;
        d->col_recv = NULL;
    }
}

/* ============================================================
 * Halo exchange
 * Grid layout: (local_rows+2) x (local_cols+2) with 1-cell halo
 * ============================================================ */
#define IDX(r, c, stride) ((r) * (stride) + (c))

static void exchange_halos(double *u, Decomp *d) {
    int lr = d->local_rows, lc = d->local_cols;
    int stride = lc + 2;
    MPI_Status st;
    MPI_Comm comm = d->cart_comm;

    /* North-South */
    if (d->north >= 0 || d->south >= 0) {
        double *send_n = &u[IDX(1, 1, stride)];
        double *recv_s = &u[IDX(lr + 1, 1, stride)];
        double *send_s = &u[IDX(lr, 1, stride)];
        double *recv_n = &u[IDX(0, 1, stride)];

        if (d->north >= 0 && d->south >= 0) {
            MPI_Sendrecv(send_n, lc, MPI_DOUBLE, d->north, 0,
                         recv_s, lc, MPI_DOUBLE, d->south, 0, comm, &st);
            MPI_Sendrecv(send_s, lc, MPI_DOUBLE, d->south, 1,
                         recv_n, lc, MPI_DOUBLE, d->north, 1, comm, &st);
        } else if (d->north >= 0) {
            MPI_Sendrecv(send_n, lc, MPI_DOUBLE, d->north, 0,
                         recv_n, lc, MPI_DOUBLE, d->north, 1, comm, &st);
        } else {
            MPI_Sendrecv(send_s, lc, MPI_DOUBLE, d->south, 1,
                         recv_s, lc, MPI_DOUBLE, d->south, 0, comm, &st);
        }
    }

    /* East-West (pack/unpack columns using pre-allocated buffers) */
    if (d->west >= 0 || d->east >= 0) {
        double *col_s = d->col_send;
        double *col_r = d->col_recv;

        if (d->west >= 0 && d->east >= 0) {
            for (int i = 0; i < lr; i++) col_s[i] = u[IDX(i+1, 1, stride)];
            MPI_Sendrecv(col_s, lr, MPI_DOUBLE, d->west, 2,
                         col_r, lr, MPI_DOUBLE, d->east, 2, comm, &st);
            for (int i = 0; i < lr; i++) u[IDX(i+1, lc+1, stride)] = col_r[i];

            for (int i = 0; i < lr; i++) col_s[i] = u[IDX(i+1, lc, stride)];
            MPI_Sendrecv(col_s, lr, MPI_DOUBLE, d->east, 3,
                         col_r, lr, MPI_DOUBLE, d->west, 3, comm, &st);
            for (int i = 0; i < lr; i++) u[IDX(i+1, 0, stride)] = col_r[i];
        } else if (d->west >= 0) {
            for (int i = 0; i < lr; i++) col_s[i] = u[IDX(i+1, 1, stride)];
            MPI_Sendrecv(col_s, lr, MPI_DOUBLE, d->west, 2,
                         col_r, lr, MPI_DOUBLE, d->west, 3, comm, &st);
            for (int i = 0; i < lr; i++) u[IDX(i+1, 0, stride)] = col_r[i];
        } else {
            for (int i = 0; i < lr; i++) col_s[i] = u[IDX(i+1, lc, stride)];
            MPI_Sendrecv(col_s, lr, MPI_DOUBLE, d->east, 3,
                         col_r, lr, MPI_DOUBLE, d->east, 2, comm, &st);
            for (int i = 0; i < lr; i++) u[IDX(i+1, lc+1, stride)] = col_r[i];
        }
    }
}

/* ============================================================
 * S1-S2 Cross-field stimulus protocol (induces spiral waves)
 * ============================================================ */
static void apply_stimulus(double *u, Decomp *d, int step) {
    int lr = d->local_rows, lc = d->local_cols;
    int stride = lc + 2;
    int N = d->N;

    /* S1: planar wave from left edge at t=0ms */
    if (step >= 0 && step < 20) {
        for (int i = 1; i <= lr; i++)
            for (int j = 1; j <= lc; j++) {
                int gc = d->col_start + (j - 1);
                if (gc < N / 20)
                    u[IDX(i, j, stride)] = 1.0;
            }
    }

    /* S2: cross-field stimulus at t=300ms to induce spiral */
    if (step >= 3000 && step < 3020) {
        for (int i = 1; i <= lr; i++)
            for (int j = 1; j <= lc; j++) {
                int gr = d->row_start + (i - 1);
                int gc = d->col_start + (j - 1);
                if (gr > N / 2 && gc < N / 2)
                    u[IDX(i, j, stride)] = 1.0;
            }
    }
}

/* ============================================================
 * Ischemic scar zone: circular region with 90% reduced diffusion
 * ============================================================ */
static double get_diffusion(int gr, int gc, int N) {
    int cr = (int)(0.6 * N), cc = (int)(0.6 * N);
    int dr = gr - cr, dc = gc - cc;
    double radius = 0.08 * N;
    if (dr * dr + dc * dc < radius * radius)
        return DIFF * 0.1;
    return DIFF;
}

/* ============================================================
 * Gather and save snapshot to binary file
 * ============================================================ */
static void save_snapshot(double *u, Decomp *d, int step) {
    int lr = d->local_rows, lc = d->local_cols;
    int stride = lc + 2;
    int N = d->N;

    double *local = (double *)calloc(lr * lc, sizeof(double));
    for (int i = 0; i < lr; i++)
        for (int j = 0; j < lc; j++)
            local[i * lc + j] = u[IDX(i+1, j+1, stride)];

    if (d->rank == 0) {
        double *global = (double *)calloc(N * N, sizeof(double));
        for (int i = 0; i < lr; i++)
            for (int j = 0; j < lc; j++)
                global[(d->row_start + i) * N + (d->col_start + j)] = local[i * lc + j];

        for (int r = 1; r < d->nprocs; r++) {
            int info[4];
            MPI_Status st;
            MPI_Recv(info, 4, MPI_INT, r, 100, d->cart_comm, &st);
            int rlr = info[2], rlc = info[3];
            double *buf = (double *)malloc(rlr * rlc * sizeof(double));
            MPI_Recv(buf, rlr * rlc, MPI_DOUBLE, r, 101, d->cart_comm, &st);
            for (int i = 0; i < rlr; i++)
                for (int j = 0; j < rlc; j++)
                    global[(info[0] + i) * N + (info[1] + j)] = buf[i * rlc + j];
            free(buf);
        }

        char fname[256];
        sprintf(fname, "results/snapshot_%06d.bin", step);
        FILE *fp = fopen(fname, "wb");
        fwrite(&N, sizeof(int), 1, fp);
        fwrite(global, sizeof(double), N * N, fp);
        fclose(fp);
        free(global);
    } else {
        int info[4] = {d->row_start, d->col_start, lr, lc};
        MPI_Send(info, 4, MPI_INT, 0, 100, d->cart_comm);
        MPI_Send(local, lr * lc, MPI_DOUBLE, 0, 101, d->cart_comm);
    }
    free(local);
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <P> %s <strategy 1|2|3> <grid_size> <num_steps> <snapshot_interval>\n", argv[0]);
            fprintf(stderr, "  Strategy: 1=Row-block, 2=Column-block, 3=2D-block\n");
            fprintf(stderr, "  Example:  mpirun -np 8 %s 3 2048 5000 2500\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int strategy = atoi(argv[1]);
    int N = atoi(argv[2]);
    int num_steps = atoi(argv[3]);
    int snap_interval = atoi(argv[4]);

    Decomp d;
    setup_decomposition(&d, strategy, N, rank, nprocs);

    if (d.rank == 0) {
        const char *names[] = {"", "Row-block", "Column-block", "2D-block"};
        printf("==============================================\n");
        printf("  Cardiac Electrophysiology Simulation\n");
        printf("  Fenton-Karma 3-Variable Ionic Model\n");
        printf("==============================================\n");
        printf("  Grid:       %d x %d  (%d cells)\n", N, N, N * N);
        printf("  Tissue:     %.1f x %.1f cm\n", N * DX, N * DY);
        printf("  Processes:  %d\n", nprocs);
        printf("  Strategy:   %s\n", names[strategy]);
        printf("  Proc grid:  %d x %d\n", d.proc_rows, d.proc_cols);
        printf("  Time steps: %d  (%.1f ms simulated)\n", num_steps, num_steps * DT);
        printf("  dx=%.4f cm  dt=%.2f ms  D=%.4f cm²/ms\n", DX, DT, DIFF);
        printf("==============================================\n");
    }

    int lr = d.local_rows, lc = d.local_cols;
    int stride = lc + 2;
    int grid_size = (lr + 2) * stride;

    double *u     = (double *)calloc(grid_size, sizeof(double));
    double *v     = (double *)malloc(grid_size * sizeof(double));
    double *w     = (double *)malloc(grid_size * sizeof(double));
    double *u_new = (double *)calloc(grid_size, sizeof(double));

    /* Resting state initialization */
    for (int i = 0; i < grid_size; i++) {
        v[i] = 1.0;
        w[i] = 1.0;
    }

    /* Local diffusion map (ischemic zone) */
    double *diff_map = (double *)malloc(lr * lc * sizeof(double));
    for (int i = 0; i < lr; i++)
        for (int j = 0; j < lc; j++)
            diff_map[i * lc + j] = get_diffusion(d.row_start + i, d.col_start + j, N);

    MPI_Barrier(d.cart_comm);
    double t_start = MPI_Wtime();

    /* ============ Main time-stepping loop ============ */
    for (int step = 0; step < num_steps; step++) {

        apply_stimulus(u, &d, step);
        exchange_halos(u, &d);

        for (int i = 1; i <= lr; i++) {
            for (int j = 1; j <= lc; j++) {
                int idx = IDX(i, j, stride);
                double uu = u[idx], vv = v[idx], ww = w[idx];

                /* Diffusion: 5-point stencil with spatially varying D */
                double D_local = diff_map[(i-1) * lc + (j-1)];
                double rx = D_local * DT / (DX * DX);
                double ry = D_local * DT / (DY * DY);
                double lap = rx * (u[IDX(i-1,j,stride)] + u[IDX(i+1,j,stride)] - 2.0*uu)
                           + ry * (u[IDX(i,j-1,stride)] + u[IDX(i,j+1,stride)] - 2.0*uu);

                /* Ionic currents */
                double du_ion, dv, dw;
                fk_step(uu, vv, ww, &du_ion, &dv, &dw);

                /* Forward Euler update */
                u_new[idx] = uu + DT * du_ion + lap;
                v[idx] = vv + DT * dv;
                w[idx] = ww + DT * dw;

                if (u_new[idx] < 0.0) u_new[idx] = 0.0;
                if (u_new[idx] > 1.0) u_new[idx] = 1.0;
            }
        }

        /* Swap */
        double *tmp = u; u = u_new; u_new = tmp;

        /* Snapshot */
        if (snap_interval > 0 && step % snap_interval == 0) {
            save_snapshot(u, &d, step);
            if (d.rank == 0)
                printf("  Step %d / %d  (t = %.1f ms)\n", step, num_steps, step * DT);
        }
    }

    MPI_Barrier(d.cart_comm);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    double max_t, min_t, avg_t;
    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, d.cart_comm);
    MPI_Reduce(&elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, d.cart_comm);
    MPI_Reduce(&elapsed, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, d.cart_comm);

    if (d.rank == 0) {
        avg_t /= nprocs;
        printf("==============================================\n");
        printf("  PERFORMANCE RESULTS\n");
        printf("==============================================\n");
        printf("  Wall time (max):  %.4f s\n", max_t);
        printf("  Wall time (min):  %.4f s\n", min_t);
        printf("  Wall time (avg):  %.4f s\n", avg_t);
        printf("  Load imbalance:   %.2f%%\n", 100.0 * (max_t - min_t) / max_t);
        printf("==============================================\n");

        FILE *csv = fopen("results/benchmark.csv", "a");
        if (csv) {
            fprintf(csv, "%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                    strategy, N, nprocs, num_steps, max_t, min_t, avg_t);
            fclose(csv);
        }
    }

    save_snapshot(u, &d, num_steps);

    free(u); free(v); free(w); free(u_new); free(diff_map);
    if (d.col_send) free(d.col_send);
    if (d.col_recv) free(d.col_recv);
    if (d.strategy == 3 && d.cart_comm != MPI_COMM_NULL)
        MPI_Comm_free(&d.cart_comm);
    MPI_Finalize();
    return 0;
}
