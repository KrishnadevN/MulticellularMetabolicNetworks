#include "fstream"
#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "string"
#include "time.h"
#include "vector"

using namespace std;

#define SAVEFILE "samples.dat"
#define r 300               // Number of variables
#define N 602               // Number of constraints
#define SAMPLETIME 10000    // Number of steps
#define samplerate 10       // Sampling rate
int timewait = 100;
#define POLYTOPE "polytope.dat"         // File: polytope
#define REACTIONS_BOUNDS "bounds.dat"   // File: bounds of the reactions
#define INITPOINT "point.dat"           // File: initial point
#define MAXFLUX 1e9     // Initialization of variables tp, tm in extrema().
                        // MAXFLUX should be larger than maximum flux
#define epsilon 1e-9
#define ID_RXN_TO_MAX 1
#define ID_RXN_TO_MAX2 0
#define MAXIMUM_EXP 20
#define EPSILON 1e-10
double BETA = 0;
double BETA2 = 0;
std::vector<int> matrice[N];        // Polytope matrix, pointers
std::vector<double> matrice2[N];    // Polytope matrix, stroichiometric coefficients
double boundmin[N];
double boundmax[N];
double flux[r];         // Polytope variables
double versori[r][r];   // Ellipsoid axis directions
double cb[r];
double cb2[r];
double ran[2];          // Random numbers
double previous[r];


double randf() {  // Random number uniform in (0,1)
    double x = double(random()) / (RAND_MAX + 1.0);
    return x;
}


void reading() {  // Read input
    fstream file;
    string nomi;
    int rev;
    file.open(REACTIONS_BOUNDS, ios::in);  // Read bounds
    for (int i = 0; i <= N - 1; i++) {
        file >> boundmin[i] >> boundmax[i];
    }
    file.close();
    file.open(POLYTOPE, ios::in);  // Read polytope
    int i1 = 0;
    int j1 = 0;
    double pepe;
    while (file >> pepe) {
        if (pepe != 0) {
            matrice[i1].push_back(j1);
            matrice2[i1].push_back(pepe);
        }
        j1++;
        if (j1 == r) {
            j1 = 0;
            i1++;
        }
    }
    file.close();
    double length;

    for (int i = 0; i <= r - 1; i++) {
        for (int j = 0; j <= r - 1; j++)
            versori[i][j] = 0;
        versori[i][i] = 1;
    }
    file.close();
    file.open(INITPOINT, ios::in);
    for (int i = 0; i <= r - 1; i++)
        file >> flux[i];
    file.close();

    int reac = ID_RXN_TO_MAX;
    for (int i = 0; i <= r - 1; i++) {  // Read objective function
        cb[i] = 0;
        for (int j = 0; j <= matrice[reac].size() - 1; j++)
            cb[i] += versori[i][matrice[reac][j]] * matrice2[reac][j];
    }
    reac = ID_RXN_TO_MAX2;
    for (int i = 0; i <= r - 1; i++) {  // Read objective function
        cb2[i] = 0;
        for (int j = 0; j <= matrice[reac].size() - 1; j++)
            cb2[i] += versori[i][matrice[reac][j]] * matrice2[reac][j];
    }
}


void findextrema(double& tp, double& tm,
                 int naxis) {  // Finds segment extrema in the direction
                               // of the axis "naxis"
    tp = MAXFLUX;
    tm = -MAXFLUX;
    for (int i = 0; i <= N - 1; i++) {
        if (matrice[i].size() > 0 && (boundmin[i] != -1000 || boundmax[i] != 1000)) {
            double x, y;
            x = y = 0;
            for (int j = 0; j <= matrice[i].size() - 1; j++) {
                x += flux[matrice[i][j]] * matrice2[i][j];
                y += versori[naxis][matrice[i][j]] * matrice2[i][j];
            }
            if (y != 0) {
                double t1, t2;
                t1 = (boundmin[i] - x) / y;
                t2 = (boundmax[i] - x) / y;
                if (t1 < 0 && t1 > tm) tm = t1;
                if (t2 < 0 && t2 > tm) tm = t2;
                if (t1 > 0 && t1 < tp) tp = t1;
                if (t2 > 0 && t2 < tp) tp = t2;
            }
        }
    }
}


void ellipsHR() {  // Hit and run with ellipsoid
    for (int i = 0; i <= r - 1; i++)
        previous[i] = flux[i];
    for (int yes = 0; yes <= r - 1; yes++) {  // Sweep over the axis directions
        double tp, tm;
        findextrema(tp, tm, yes);
        double t = 0;
        int count = 0;
        do {
            double c = BETA * cb[yes] + BETA2 * cb2[yes];
            double ranvar = randf();
            if (fabs(c) < EPSILON)
                t = tm + (tp - tm) * ranvar;
            else {
                if (fabs(c * (tp - tm)) < MAXIMUM_EXP)
                    t = tm + log(1. + ranvar * (exp(c * (tp - tm)) - 1)) / c;
                else {
                    if (c < 0)
                        t = tm + log(1 - ranvar) / c;
                    else
                        t = tp + log(ranvar) / c;
                }
            }
            count++;
        } while ((t - tm < EPSILON || tp - t < EPSILON) && count < 10);
        if (count == 10)
            t = 0;
        for (int i = 0; i <= r - 1; i++)
            flux[i] += t * versori[yes][i];
    }
    int oko = 1;
    for (int i = 0; i <= N - 1; i++) {
        double flux1 = 0;
        if (matrice[i].size() > 0)
            for (int j = 0; j <= matrice[i].size() - 1; j++)
                flux1 += matrice2[i][j] * flux[matrice[i][j]];
        if (flux1 < boundmin[i] || flux1 > boundmax[i])
            oko = 0;
    }
    if (oko == 1)
        for (int i = 0; i <= r - 1; i++)
            previous[i] = flux[i];
    else
        for (int i = 0; i <= r - 1; i++)
            flux[i] = previous[i];
}


void minover() {  // Relaxation algorithm to find a point inside
    int ok = 0;
    double alpha;
    double xmin;

    do {
        int min;
        int sign;
        xmin = 100000000000;
        for (int i = 0; i <= N - 1; i++) {
            if (matrice[i].size() > 0) {
                double x = 0;
                for (int j = 0; j <= matrice[i].size() - 1; j++)
                    x += flux[matrice[i][j]] * matrice2[i][j];
                double x1 = x - boundmin[i];
                double x2 = boundmax[i] - x;
                if (x1 < xmin) {
                    xmin = x1;
                    min = i;
                    sign = 1;
                }
                if (x2 < xmin) {
                    xmin = x2;
                    min = i;
                    sign = -1;
                }
            }
        }
        if (xmin > 0)
            ok = 1;
        else {
            double norm = 0;
            for (int j = 0; j <= matrice[min].size() - 1; j++)
                norm += matrice2[min][j] * matrice2[min][j];
            alpha = -1.8 * xmin / norm;
            if (alpha < 1e-64) alpha = 1e-64;
            for (int j = 0; j <= matrice[min].size() - 1; j++)
                flux[matrice[min][j]] += sign * alpha * matrice2[min][j];
        }
    } while (ok == 0);
}


int main(int argc, char** argv) {
    srand(time(0));  // Seed for random numbers
    reading();       // Read input
    minover();

    double BETA_G = -11.75;
    double BETA_O = -1.65;
    int NPOINTS = 200;
    string filename = SAVEFILE;

    if (argc == 3) {
        BETA_G = atof(argv[1]);
        BETA_O = atof(argv[2]);
    } else if (argc == 4) {
        BETA_G = atof(argv[1]);
        BETA_O = atof(argv[2]);
        NPOINTS = atoi(argv[3]);
    } else if (argc == 5) {
        BETA_G = atof(argv[1]);
        BETA_O = atof(argv[2]);
        NPOINTS = atoi(argv[3]);
        filename = argv[4];
    }

    cout << "Sampling " << NPOINTS << " points with parameters:\n";
    cout << "\tBETA_G = " << BETA_G << endl;
    cout << "\tBETA_O = " << BETA_O << endl;
    cout << "Saving data to file: " << filename << endl;

    ofstream myfile(filename);

    BETA = -BETA_G;  // Glucose, flipped sign (positive instead of negative)
    BETA2 = BETA_O;  // Oxygen

    int points = 0;
    double previous[r];
    for (int i = 0; i <= r-1; i++)
        previous[i] = flux[i];
    for (int i = 0; i <= SAMPLETIME + int(timewait); i++) { // Sampling
        ellipsHR();
        if (i > timewait && i % samplerate == 0) {
            for (int n = 0; n <= r/2 - 1; n++)
                myfile << flux[n + r/2] << " ";             // Save glucose
            myfile << endl;
            for (int n = 0; n <= r/2 - 1; n++)
                myfile << flux[n] << " ";                   // Save oxygen
            myfile << endl;
            for (int n = 0; n <= r/2 - 1; n++) {
                double lac = 2.*flux[n + r/2] - flux[n]/3.; // Save lactate
                myfile << lac << " ";
            }
            myfile << endl;
            points++;
            cout << "\rPoints saved: " << points << "/" << NPOINTS << "\t" << flush;
        }
        if (points >= NPOINTS)
            break;
    }
    myfile.close();
    cout << endl;
    return 0;
}
