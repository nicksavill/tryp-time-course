#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "mkl_vsl.h"
#include "omp.h"

#define MIN(A, B) (A) < (B)? (A) : (B) // return minimum of A and B

// possible states of tryps
enum tryp_state {Empty, Not_Viable, Viable};

// an individual tryp data structure
typedef struct {
    // an array containing the copy number of each minicircle class in a tryp
    unsigned short* class;
    // the status of a tryp
    // whether it exists or not
    // and if it exists whether it is viable or not
    // ie. has less than the maximum number of minicircles (implemented)
    // and has all gRNAs to fully edit required mRNAs (not implemented) 
    enum tryp_state state;
} tryp_t;

// simulation parameters data structure
typedef struct {
    unsigned int maxnumminicircles; // maximum number of minicircles in a network
    unsigned int numclasses;        // number of minicircle classes
    unsigned int generations;       // number of generations
    unsigned int pop_bottleneck;    // population after bottleneck
    unsigned int pop_threshold;     // population threshold for bottlenecking
    unsigned int pop;               // current number of tryps
    unsigned int rseed;             // seed for random number generator
    unsigned int nthreads;          // number of parallel threads
    bool all_required;              // whether all minicircle classes are required or not
    bool load;                      // whether to load a previous simulation
    char savefile[100];             // filename of saved simulation
    char loadfile[100];             // filename of saved simulation to load
    char avecopynumfile[100];       // filename of pop aveerage class copy number
    double p;                       // the probability of segregation
} sim_pars_t;

// function declarations
void Run_sim(tryp_t*, sim_pars_t*, VSLStreamStatePtr*);
void Next_Generation(tryp_t*, sim_pars_t*, VSLStreamStatePtr, int, int);
void Bottleneck(tryp_t*, sim_pars_t*, int, VSLStreamStatePtr);
void Save_viable_tryps(tryp_t*, sim_pars_t*);
void Initialise_population(tryp_t*, sim_pars_t*);
void Pop_average_class_copy_number(tryp_t*, sim_pars_t*);
int Count_viable_tryps(tryp_t*, sim_pars_t*);
int binomialRNG(int, double, VSLStreamStatePtr);
int uniformRNG(int, VSLStreamStatePtr);


int main(int argc, char* argv[])
{
    unsigned int i;
    tryp_t* tryps;              // each trypanosome in the population
    sim_pars_t pars;            // the simulation parameters
    VSLStreamStatePtr* streams; // random number streams, one for each thread

    pars.generations = 50;
    pars.pop = 100;
    pars.pop_threshold = 5000;
    pars.pop_bottleneck = 10;
    pars.numclasses = 398;
    pars.maxnumminicircles = 5000;
    pars.all_required = false;
    pars.p = 0.5;
    pars.rseed = 1;
    pars.nthreads = omp_get_num_threads();
    pars.load = false;
    strcpy(pars.savefile, "save1.txt");
    strcpy(pars.loadfile, "save.txt");
    strcpy(pars.avecopynumfile, "avecopynum.csv");

    // setup RNG streams for for each parallel thread
    streams = (VSLStreamStatePtr*) malloc(pars.nthreads * sizeof(VSLStreamStatePtr)); 
    for (i = 0; i < pars.nthreads; i++)
        vslNewStream(&streams[i], VSL_BRNG_MRG32K3A, pars.rseed+i);
  
    //initialise memory to store copy number of each minicircle class in each tryp 
    tryps = (tryp_t*) malloc(2*pars.pop_threshold * sizeof(tryp_t));
    for (i = 0; i < 2*pars.pop_threshold; i++)
        tryps[i].class = (unsigned short*) malloc(pars.numclasses * sizeof(unsigned short));
  
    // initialise the starting population of tryps
    Initialise_population(tryps, &pars);

    // run the simulation
    Run_sim(tryps, &pars, streams);

    // save final population average class copy numbers
    Pop_average_class_copy_number(tryps, &pars);

    // save final viable tryp population
    Save_viable_tryps(tryps, &pars);

    // free tryps memory
    for (i = 0; i < 2*pars.pop_threshold; i++)
        free(tryps[i].class);
    free(tryps);

    // free RNG streams memory
    for (i = 0; i < pars.nthreads; i++)
        vslDeleteStream(&streams[i]);
    return 0;
}

void Initialise_population(tryp_t* tryps, sim_pars_t* pars) {
    // if pars->load is true then tryps will be initialised from a file called loadfile
    // the population size is set to the number of tryps in the loadfile
    // if pars->load is false then all minicircle classes in all tryps are initialised
    // with the same value of maxnumminicircles / numclasses rounded down so that
    // network size will not be equal to maxnumminicircles
    unsigned int i, j;

    if (pars->load) {
        // load each tryp's copy number from a saved file
        FILE *fp = fopen(pars->loadfile, "r");

        fscanf(fp, "n=%d", &pars->pop);

        for (i = 0; i < pars->pop; i++) {
            tryps[i].state = Viable; 
            for (j = 0; j < pars->numclasses; j++)
                fscanf(fp, "%hd", &tryps[i].class[j]);
        }
        fclose(fp);
    }
    else {
        // all classes in all tryps have equal copy number
        for (i = 0; i < pars->pop; i++) {
            tryps[i].state = Viable;
            for (j = 0; j < pars->numclasses; j++)
                tryps[i].class[j] = pars->maxnumminicircles / pars->numclasses;
        }
    }
}

void Run_sim(tryp_t* tryps, sim_pars_t* pars, VSLStreamStatePtr* streams) {
    unsigned int i, j, t, n, iam, ipoints, istart;

    n = Count_viable_tryps(tryps, pars);
    printf("%d\t%d\t%d\n", 0, pars->pop, n);

    // run the simulation for a number of generations
    for (t = 1; t <= pars->generations; t++) {
        // perform a bottleneck if population size is greater than the threshold
        if (pars->pop > pars->pop_threshold)
            Bottleneck(tryps, pars, n, streams[0]);

        // split the calculation of tryp divisions across parallel threads
#       pragma omp parallel default(shared) private(iam, nt, ipoints, istart)
        {
            iam = omp_get_thread_num();
            ipoints = pars->pop / pars->nthreads;
            istart = iam * ipoints;
            if (iam == pars->nthreads-1) 
                ipoints = istart + pars->pop; //set the part of the array that each thread can copy into
            Next_Generation(tryps, pars, streams[iam], istart, ipoints);
        }

        n = Count_viable_tryps(tryps, pars);
        if (n == 0) {
            printf("No viable tryps. Exiting simulation.\n");
            return;
        }
        else
            printf("%d\t%d\t%d\n", t, pars->pop, n);
    }
}

void Save_viable_tryps(tryp_t* tryps, sim_pars_t* pars) {
    // save all viable tryps to file
    int i, j, n = Count_viable_tryps(tryps, pars);
    FILE *fp = fopen(pars->savefile, "w");

    fprintf(fp, "n=%d\n", n);

    for (i = 0; i < pars->pop; i++)
        if (tryps[i].state == Viable) { 
            for (j = 0; j < pars->numclasses; j++)
                fprintf(fp, "%hd ", tryps[i].class[j]);
            fprintf(fp,"\n");
        }
   fclose(fp);
}

int Count_viable_tryps(tryp_t* tryps, sim_pars_t* pars) {
    // count number of viable tryps in current population
    int i, count = 0;

    for (i = 0; i< pars->pop; i++)
        if (tryps[i].state == Viable) 
            count++;
    return count;
}

void Next_Generation(tryp_t* tryps, sim_pars_t* pars, VSLStreamStatePtr stream, int istart, int ipoints) {
    unsigned int i, j, r;
    unsigned short total_minicircles, n, m, s, o, a;
    unsigned short leftover_in_class[pars->numclasses];
    bool any_class_missing = false;

    // on this thread update tryps istart through istart+ipoints-1
    for (i = istart; i < istart+ipoints; i++) {
        // divide parent tryp i (if it is viable) into two daugther tryps with
        // indicies i and r 
        r = i + pars->pop;

        if (tryps[i].state == Viable) {
            // tryp i divides as it exists and is viable

            // set total minicircles in daughter i to 0
            total_minicircles = 0;

            // loop through each minicircle class and segregate minicircles into daughter tryp i
            for (j = 0; j < pars->numclasses; j++) {

                // n is the copy number of minicircle class j in parent i before
                // DNA replication. there are 2n minicircles after DNA replication
                n = tryps[i].class[j];
        
                /* Each minicircle is replicated creating 2 sibling minicircles. The 1st of
                the two siblings has a 50% chance of ending up in daughter tryp i and
                a 50% chance of ending up in daughter tryp r. Given that there are n
                copies of minicircle class j, there are n 1st siblings and n 2nd siblings
                of class j. Therefore the number of 1st siblings m, ending up in daughter
                tryp i is binomially distributed with parameters n and 0.5. This means
                the number of 1st siblings ending up in daughter tryp r is n-m. 
                */
                m = binomialRNG(n, 0.5, stream);

                /* If the 1st sibling ended up in daughter tryp i, the probability of the
                2nd sibling ending up in daughter tryp i is 1-p. (p is the segregation
                probability. If p=0 the two siblings end up in the same daughter tryp,
                and if p=1 they end up in different daughter tryps.) Given that m 1st
                siblings ended up in daughter tryp i, the number of 2nd siblings s,
                ending up in daughter tryp i is binomially distributed with parameters m
                and 1-p. 
                */
                s = binomialRNG(m, 1. - pars->p, stream);

                /* Conversely, if the 1st sibling ended up in daughter tryp r, the
                probability of the 2nd sibling ending up in daughter tryp i is p. Given
                that n-m 1st siblings ended up in daughter tryp r (see above), the number
                of 2nd siblings o, ending up in daughter tryp i is binomially distributed
                with parameters n-m and p. 
                */
                o = binomialRNG(n - m, pars->p, stream);

                // total number of minicircles (1st and 2nd siblings) ending up in 
                // daughter tryp i
                a = m+s+o;
                tryps[i].class[j] = a;
	
                // increment the network size of daughter tryp i
                total_minicircles += a;

                // daughter tryp r (see below) gets the rest of the copies of minicircle class j
                leftover_in_class[j] = 2*n-a;

                // check if any minicircle class is missing
                any_class_missing = any_class_missing | (a == 0);
            }

            // make tryp i inviable if network size is above maxnumminicircles
            // or any class missing if all required
            if (total_minicircles > pars->maxnumminicircles || (pars->all_required & any_class_missing)) 
                tryps[i].state = Not_Viable;

            // now do daughter tryp r
            total_minicircles = 0;
            any_class_missing = false;

            for (j = 0; j < pars->numclasses; j++) {
                // daughter tryp r gets this many copies
                a = leftover_in_class[j];
                tryps[r].class[j] = a;

                // increment the network size of daughter tryp r
                total_minicircles += a;

                // check if any minicircle class is missing
                any_class_missing = any_class_missing | (a == 0);

            }

            // set viability of daughter tryp r
            if (total_minicircles > pars->maxnumminicircles || (pars->all_required & any_class_missing)) 
                tryps[r].state = Not_Viable;
            else
                tryps[r].state = Viable;
        }
        else
            // set state of daughter r to Empty if daughter i is inviable or does not exist
            tryps[r].state = Empty;
    }

    // double the current population size
    pars->pop *= 2;
}

void Bottleneck(tryp_t* tryps, sim_pars_t* pars, int nviable, VSLStreamStatePtr stream) {
    unsigned int i, r, j, numbertokeep; 
  
    // number of tryps to keep at bottleneck
    // if number viable is less than pop_bottleneck use nviable
    numbertokeep = MIN(nviable, pars->pop_bottleneck);

    // create temporary storage for kept tryps
    tryp_t tmp_tryps[numbertokeep];
    for (i = 0; i < numbertokeep; i++)
        tmp_tryps[i].class = (unsigned short*) malloc(pars->numclasses * sizeof(unsigned short));

    // randomly choose viable tryps from the current populationto be kept in the bottlenecked population 
    i = 0; 
    while (i < numbertokeep) {
        // randomly chose a tryp from the current population
        r = uniformRNG(pars->pop, stream); 

        // check that it is viable
        if (tryps[r].state == Viable) { 
            // if it is, store its classes in the temporary storage
            for (j = 0; j < pars->numclasses; j++)
                tmp_tryps[i].class[j] = tryps[r].class[j]; 

            // set state of choosen tryp to not exist so it cannot be picked again
            tryps[r].state = Empty;

            // increment the number of viable chosen tryps 
            i++;
        }
    }

    // set the new population size
    pars->pop = numbertokeep;

    // copy temporary storage into tryps
    for (i = 0; i < numbertokeep; i++) {
        tryps[i].state = Viable;
        for (j = 0; j < pars->numclasses; j++)
            tryps[i].class[j] = tmp_tryps[i].class[j];
    } 

    // free storage
    for (i = 0; i < numbertokeep; i++)
        free(tmp_tryps[i].class);
}


void Pop_average_class_copy_number(tryp_t* tryps, sim_pars_t* pars) {
    // the population average copy number of each minicircle class
    int i, j, n = Count_viable_tryps(tryps, pars);
    double* count = (double*) calloc(pars->numclasses, sizeof(double));
  
    // count number of each minicircle class in each viable tryp
    for (i = 0; i < pars->pop; i++)
        if (tryps[i].state == Viable)
            for (j = 0; j < pars->numclasses; j++)
                count[j] += tryps[i].class[j];

    FILE *fp = fopen(pars->avecopynumfile, "w");
    fprintf(fp, "class,avecopynum\n");
    
    for (j = 0; j < pars->numclasses; j++)
        fprintf(fp, "%d,%e\n", j, count[j] / n);
    fclose(fp);

    free(count);
}

int binomialRNG(int n, double p, VSLStreamStatePtr stream) {
    // return a single binomially distributed random number
    if (n == 0)
        return 0;
    int r;
    if (viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 1, &r, n, p)) {
        fprintf(stderr, "Error in binomiaRNG n=%d p=%e\n", n, p);
        exit(0);
    }
    return r;
}

int uniformRNG(int a, VSLStreamStatePtr stream) {
    int r;
    if (viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &r, 0, a)) {
        fprintf(stderr, "Error in uniformRNG a=%d\n", a);
        exit(0);
    }
    return r;
}

