#include <iostream>
#include <fstream>
#include <sstream>
#include "mpi.h"
#include <math.h>

const double a = 1;
const double xMin = 0, xMax = 10, tMax = 3.0;
const int M = 1000, K = 1000;
double tau = tMax/(K-1), h = (xMax - xMin)/(M-1);

/// Starting condition.
double phi(double x)
{
    return sin(M_PI*x/(xMax-xMin));
}

/// Border condition.
double psi(double t)
{
    return 1;
}

/// Source term.
double f(double t, double x)
{
    return 0;
//    double sigma = 0.1;
//    double value = (x/sigma)*(x/sigma)/2;
//    return (1 - value)*exp(-value)/(M_PI*pow(sigma, 4));
}

int getStart(int rank, int portion)
{
    return rank*portion;
}

int getEnd(int rank, int portion)
{
    int end = (rank+1)*portion;
    return (end > M) ? M : end;
}

void saveData(std::ofstream &file, double *all)
{
    for (int m = 0; m < M; m++)
        file << all[m] << ";";
    file << "\n";
}

void updateBorders(int rank, int size, int portion, double *all, double *valueLeft, double *valueRight)
{
    if (rank == 0)
    {
        /// Send border values for different processes (each process has own part of physical space to compute).
        for (int i = 1; i < size-1; i++)
        {
            *valueLeft = all[i*portion-1];
            *valueRight = all[(i+1)*portion];
            MPI_Send(valueLeft, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(valueRight, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        int lastIndex = (M % size == 0) ? (M-portion-1) : (M - (M%size) - 1);
        MPI_Send(&all[lastIndex], 1, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD);
        *valueLeft = 0; *valueRight = all[portion];
    }
    else
    {
        /// Receive border values in others.
        MPI_Recv(valueLeft, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank != size - 1)
            MPI_Recv(valueRight, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char* argv[])
{
    std::stringstream text(std::stringstream::out | std::stringstream::binary);
    std::ofstream file;
    file.open("../res/out.dat", std::ofstream::binary);
    if (!file.is_open())
        return -1;
    auto all = new double[M];
    for (int i = 0; i < M; i++)
        all[i] = 0;

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /// Data arrays.
    double *tmp = nullptr;
    int portion = M/size;
    if (M % size)
        portion += 1;

    auto up1 = new double[portion];
    auto u = new double[portion];
    auto um1 = new double[portion];

    int start = getStart(rank, portion);
    int end = getEnd(rank, portion);

    /// Setup (zero step).
    for (int m = start; m < end; m++)
        u[m-start] = phi(m*h);

    MPI_Gather(&u[0], end-start, MPI_DOUBLE, all, end-start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        saveData(file, all);

    double valueLeft = 0, valueRight = 0;
    updateBorders(rank, size, portion, all, &valueLeft, &valueRight);

    /// First step.
    for (int m = start; m < end; m++)
    {
        double left = valueLeft, right = valueRight;
        if (m > start) left = u[m - 1 - start];
        if (m < end-1) right = u[m + 1 - start];

        if (m == 0)
            up1[0] = psi(1 * tau);
        else if (m == M-1)
            up1[M - 1 - start] = u[M - 1 - start] + tau * f(1 * tau, (M - 1) * h) - a * tau / h * (u[M - 1 - start] - u[M - 2 - start]); // corner
        else
            up1[m - start] = (right + left) / 2 + tau * f(1 * tau, m * h) - a * tau / h * (right - left) / 2; // explicit three-point scheme
    }

    /// Swap buffers.
    MPI_Gather(&up1[0], end-start, MPI_DOUBLE, all, end-start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        saveData(file, all);
    tmp = um1; um1 = u; u = up1; up1 = tmp; tmp = nullptr;

    /// Loop over time, start form second point.
    for (int k = 2; k < K; k++)
    {
        updateBorders(rank, size, portion, all, &valueLeft, &valueRight);
        for (int m = start; m < end; m++)
        {
            double left = valueLeft, right = valueRight;
            if (m > start) left = u[m - 1 - start];
            if (m < end-1) right = u[m + 1 - start];

            if (m == 0)
                up1[0] = psi(k * tau);
            else if (m == M-1)
                up1[M - 1 - start] = u[M - 1 - start] + tau * f(1 * tau, (M - 1) * h) - a * tau / h * (u[M - 1 - start] - u[M - 2 - start]); // corner
            else
                up1[m - start] = um1[m - start] + 2 * tau * f(k * tau, m * h) - a * tau / h * (right - left); // cross
        }

        /// Swap buffers.
        MPI_Gather(&up1[0], end-start, MPI_DOUBLE, all, end-start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            saveData(file, all);
        tmp = um1; um1 = u; u = up1; up1 = tmp; tmp = nullptr;
    }

    if (rank == 0)
    {
        file.write(text.str().c_str(), text.str().length());
        file.close();
        delete[] all;
    }

    delete[] u;
    delete[] up1;
    delete[] um1;

    MPI_Finalize();
    return 0;
}
