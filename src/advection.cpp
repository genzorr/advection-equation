#include <iostream>
#include <fstream>
#include <sstream>
#include "mpi.h"
#include <math.h>

const double a = 1;
const double xMin = 0, xMax = 1.0, tMax = 3.0;
const int M = 100, K = 1000;
double tau = tMax/(K-1), h = (xMax - xMin)/(M-1);

/// Starting condition.
double phi(double x)
{
    return sin(M_PI*x/(xMax-xMin));
}

/// Border condition.
double psi(double t)
{
    return 0;
}

/// Source term.
double f(double t, double x)
{
    double sigma = 0.1;
    double value = (x/sigma)*(x/sigma)/2;
    return (1 - value)*exp(-value)/(M_PI*pow(sigma, 4));
}

int main()
{
    std::stringstream text(std::stringstream::out | std::stringstream::binary);
    std::ofstream file;
    file.open("../res/out.dat", std::ofstream::binary);
    if (!file.is_open())
        return -1;

    /// Data arrays.
    auto up1 = new double[M];
    auto u = new double[M];
    auto um1 = new double[M];

    /// Setup (zero step).
    for (int m = 0; m < M-1; m++)
    {
        u[m] = phi(m*h);
        text << u[m] << ";";
    }
    u[M-1] = phi((M-1)*h);
    text << u[M-1] << "\n";

    /// First step.
    up1[0] = psi(1*tau);
    text << up1[0] << ";";
    for (int m = 1; m < M-1; m++)
    {
        up1[m] = (u[m+1]+u[m-1])/2 + tau*f(1*tau,m*h) - a*tau/h*(u[m+1]-u[m-1])/2; // explicit three-point scheme
        text << up1[m] << ";";
    }
    up1[M-1] = u[M-1] + tau*f(1*tau,(M-1)*h) - a*tau/h*(u[M-1] - u[M-2]); // corner
    text << up1[M-1] << "\n";

    /// Swap buffers.
    double *tmp = um1;
    um1 = u; u = up1; up1 = tmp; tmp = nullptr;

    /// Loop over time, start form second point.
    for (int k = 2; k < K; k++)
    {
        up1[0] = psi(k*tau);
        text << up1[0] << ";";
        for (int m = 1; m < M-1; m++)
        {
            up1[m] = um1[m] + 2*tau*f(k*tau, m*h) - a*tau/h*(u[m+1]-u[m-1]); // cross
            text << up1[m] << ";";
        }
        up1[M-1] = u[M-1] + tau*f(1*tau,(M-1)*h) - a*tau/h*(u[M-1] - u[M-2]); // corner
        text << up1[M-1] << "\n";
        tmp = um1, um1 = u; u = up1; up1 = tmp; tmp = nullptr;
    }

    delete[] u;
    delete[] up1;
    delete[] um1;

    file.write(text.str().c_str(), text.str().length());
    return 0;
}
