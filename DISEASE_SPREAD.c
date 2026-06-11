
/* 

TO COMPILE: 
mpiCC DISEASE_SPREAD.c -I/usr/include/allegro5 -L/usr/lib -lallegro -lallegro_primitives

TO RUN:
mpirun -np X ./a.out      

X IS THE NUMBER OF PROCESSES, example: mpirun -np 2 ./a.out

*/


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

// HOW MANY COLS AND ROWS IN THE WHOLE CA?

#define numCols 300 // 300
#define numRows 300 // 300


int NCOLS = numCols;
int NROWS = numRows; 


int r_rows;
int r_cols;


int nsteps = 800;  // 800

#define v(r,c) ((r)*NCOLS+(c))

#define m(r,c) ((r)*(r_cols*dims[1])+(c)) // macro for the matrix on process rank 0 that contains the results, used in 2D

typedef struct{

    int status;
    int T;

    bool Im;     // if true, the person can become immune

} Person;


Person* readM;
Person* writeM;
Person* Mat;

int rank;
int processes;

int idM = 100;   // ID OF A PERSON WHO'S COMPLETELY SICK
int cInf = 20;   // INFECTIVE CONSTANT, AN INFECTED PERSON IS ADDED THIS EACH STEP, TO SIMULATE THE DEVELOPMENT OF THE DISEASE

int k1 = 3;      // INFECTION CONSTANTS, THE BIGGER THEY ARE THE HARDER IT IS TO INFECT A HEALTHY PERSON
int k2 = 3;      //

int soglia = nsteps - (nsteps - 40);       // IF A PERSON TAKES THE DISEASE 'SOGLIA' TIMES (AKA THE T MEMBER OF THE STRUCT THAT WORKS AS COUNTER) THEN THE PERSON DEVELOPS ANTIBODIES AND CANNOT TAKE THE DISEASE ANYMORE
                                           // THE NUMBER SUBTRACTED BY THE nsteps INSIDE THE PARENTHESIS: THE SMALLER IT IS, THE QUICKER THE PERSON DEVELOPS THE ANTIBODIES

bool check = false;            // CHECKS IF IT'S A 2D GRID (IF FALSE, IT'S NOT)

MPI_Datatype PersonType;       // NEW DATATYPE FOR STRUCTS

MPI_Datatype contiguousRow;    // DATATYPE FOR ROW (1D)

MPI_Datatype datatype_row;     // DATATYPE FOR ROW (2D)
MPI_Datatype datatype_column;  // DATATYPE FOR COLUMN (2D)

MPI_Datatype subMatrix;        // DATATYPE TO SEND SUBMATRIX (2D)
MPI_Datatype Matrix;           // DATATYPE TO RECEIVE THE SUBMATRIX (2D)






MPI_Comm New_Comm;            // 1D AND 2D COMMUNICATOR

int up;      // FOR THE CARTESIAN SHIFT (1D - 2D)
int down;

int left;
int right;


int dims[2] = {0,0};   // HOW THE GRID IS FORMED
int MAP[3][3];         // NEIGHBOURS MAP


int rsize;
int csize;



int cellSize = 3;  // 3   GRAPHICAL SIZE OF CELLS


// FUNCTIONS

void init();
void SequentialInit();
inline void Initial();
void print(int &step);
void swap(Person * &readM, Person * &writeM);
inline void transFuncCell(int i, int j);
inline void transFuncBorder();
inline void transFuncInside();
inline void transFunc();
inline void receivePropAndWork(int &s);
inline void sendPropAndWork(int &s);
void BDwork();
void BDtransFuncNoBorder();
void BDtransFuncOnBorder();
void drawInterface();



int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);     
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (processes > 1){
 
        MPI_Dims_create(processes, 2, dims);    // dims[0] ha il numero di righe, dims[1] quello di clonne

        if ( (dims[1] == 1) ){     // 1d
            
            int ndims = 1; // number of dimensions, 1 if 1d or 2 if 2d       PRIMA LE Y RIGHE E POI LE X COLONNE, SU OGNI COSA, PERIODS ETC..
            int periods[1] = {1}; // false or true for each dimension, if true it's periodic on the dimension n 
            int dim[1] = {processes}; // number of processes on each dimension
            int reorder = 0; // false or true if it's allowed to assign a new rank to each process


            MPI_Cart_create(MPI_COMM_WORLD, ndims, dim, periods, reorder, &New_Comm);
    
            MPI_Cart_shift(New_Comm, 0, 1, &up, &down);

            //printf("Le dimensioni sono: %d %d",dims[0], dims[1]);
        }

        else{     // 2D

            check = true;

            int ndims = 2;  // 2D
            int periods[2] = {1,1};   // toroidal
            int reorder = 0;

            MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &New_Comm);

            int my_coordinates[2];    // my_coordinates[0] è il numero della riga, my_coordinates[1] quello della colonna
            MPI_Cart_coords(New_Comm, rank, ndims, my_coordinates);

            int cont[2] = {0,0};


            MPI_Cart_shift(New_Comm, 0, 1, &up, &down);
            MPI_Cart_shift(New_Comm, 1, 1, &left, &right);


            int left_coord[2];
            int right_coord[2];


            MPI_Cart_coords(New_Comm, left, ndims, left_coord);
            MPI_Cart_coords(New_Comm, right, ndims, right_coord);


            MAP[1][1] = rank;
            MAP[1][0] = left;
            MAP[1][2] = right;
            MAP[0][1] = up;
            MAP[2][1] = down;

            // NOW WE NEED TO FIND OUR DIAGONAL NEIGHBORS


            int y, x;

            y = my_coordinates[0];   // my row position
            x = my_coordinates[1];   // my column position

            int yy, xx;

            int t_coords[2];


            // TOP LEFT

            if ( y - 1 < 0){             
                yy = dims[0] - 1;
            }

            else{
                yy = y - 1;
            }


            if ( x - 1 < 0){
                xx = dims[1] - 1;
            }

            else{
                xx = x -1;
            }

            t_coords[0] = yy;
            t_coords[1] = xx;

            MPI_Cart_rank(New_Comm, t_coords, &MAP[0][0]);



            // TOP RIGHT


            if ( y - 1 < 0){
                yy = dims[0] - 1;
            }

            else{
                yy = y - 1;
            }


            if ( x + 1 >= dims[1]){
                xx = 0;
            }

            else{
                xx = x + 1;
            }

            t_coords[0] = yy;
            t_coords[1] = xx;

            MPI_Cart_rank(New_Comm, t_coords, &MAP[0][2]);



            // BOTTOM LEFT


            if ( y + 1 >= dims[0]){
                yy = 0;
            }

            else{
                yy = y + 1;
            }


            if ( x - 1 < 0){
                xx = dims[1] - 1;
            }

            else{
                xx = x -1;
            }

            t_coords[0] = yy;
            t_coords[1] = xx;

            MPI_Cart_rank(New_Comm, t_coords, &MAP[2][0]);



            // BOTTOM RIGHT
 

            if ( y + 1 < 0){
                yy = 0;
            }

            else{
                yy = y + 1;
            }


            if ( x + 1 >= dims[1]){
                xx = 0;
            }

            else{
                xx = x + 1;
            }


            t_coords[0] = yy;
            t_coords[1] = xx;

            MPI_Cart_rank(New_Comm, t_coords, &MAP[2][2]);

            //printf("I'm process of rank %d, dimension is %d %d, my coordinates are: %d e %d, TOP LEFT: %d,TOP CENTER: %d, TOP RIGHT: %d, LEFT: %d, CENTER: %d, RIGHT %d, BOTTOM LEFT: %d, BOTTOM CENTER: %d, BOTTOM RIGHT: %d            ", rank, dims[0], dims[1], my_coordinates[0], my_coordinates[1], MAP[0][0], MAP[0][1], MAP[0][2], MAP[1][0], MAP[1][1], MAP[1][2], MAP[2][0], MAP[2][1], MAP[2][2]);
            
    
        }
    }




    if (processes > 1){

        int count = 2;  // Length of following arrays
        int array_blocklengths[count] = {2,1};  // How many elements of each block
        MPI_Datatype array_types[count] = {MPI_INT,MPI_C_BOOL};  // Array of MPI_Datatypes of each block
        MPI_Aint array_displacements[count] = {offsetof(Person, status), offsetof(Person, Im)};  // Array of displacements of each block

        MPI_Type_create_struct(count, array_blocklengths, array_displacements, array_types, &PersonType);
        MPI_Type_commit(&PersonType);
    }


    
    double starttime = MPI_Wtime();



    if ( (processes == 1) ){        // SEQUENTIAL
        
        al_init();
        al_init_primitives_addon();

        ALLEGRO_DISPLAY *display;

        display = al_create_display(NCOLS * cellSize + 20, NROWS * cellSize + 20);

        if ( display == NULL ){
            printf("Display error");
        }
        al_set_window_title(display, "DISEASE SPREAD");

        readM = new Person[NROWS*NCOLS];
        writeM = new Person[NROWS*NCOLS];
        SequentialInit();

        for(int s=0;s<nsteps;s++){
            drawInterface();
            //al_rest(0.3);
            transFunc();
            //print(s);
            swap(readM,writeM);
        }

        al_destroy_display(display);
        al_uninstall_system();

    }


    if ( (processes > 1) && (check == false) ){     // PARALLEL 1D


        MPI_Type_contiguous(NCOLS, PersonType, &contiguousRow);

        MPI_Type_commit(&contiguousRow);


        if ( (rank == 0 ) ){      // FIRST PROCESS, PARALLEL  1D

            al_init();
            al_init_primitives_addon();

            ALLEGRO_DISPLAY *display;

            display = al_create_display(NCOLS * cellSize + 20, NROWS * cellSize + 20);
            
            if ( display == NULL ){
                printf("Display error");
            }
            al_set_window_title(display, "DISEASE SPREAD");
            


            readM = new Person[(NROWS/processes +2)*NCOLS];
            writeM = new Person[(NROWS/processes +2)*NCOLS];
            Mat = new Person[NROWS*NCOLS];
            init();


            for(int s=0; s<nsteps; s++){
                receivePropAndWork(s);
                drawInterface();
                //al_rest(0.3);
                //print(s);
                swap(readM,writeM);
            }

            al_destroy_display(display);
            al_uninstall_system();
        }




        else if ( (rank != 0) && (rank != processes-1) ){          // MIDDLE PROCESSES, PARALLEL  1D                           
            readM = new Person[(NROWS/processes +2)*NCOLS];
            writeM = new Person[(NROWS/processes +2)*NCOLS];
            init();
            for(int s=0;s<nsteps;s++){
                sendPropAndWork(s);
                swap(readM,writeM);
            }
        }

        else if ( (rank == (processes - 1)) ){       // LAST PROCESS, PARALLEL  1D
            readM = new Person[((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes ))+2)*NCOLS];
            writeM = new Person[((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes ))+2)*NCOLS];
            init();
            for(int s=0;s<nsteps;s++){
                sendPropAndWork(s);
                swap(readM,writeM);
            }
        }

    }







    if ( (processes > 1) && (check == true) ){


        r_rows = numRows/dims[0];     // number of "real" rows
        r_cols = numCols/dims[1];     // number of "real" columns

        NROWS = r_rows + 2;
        NCOLS = r_cols + 2;

        int count = r_cols;

        MPI_Type_contiguous(count, PersonType, &datatype_row);     // datatype for rows


        int blockcount = r_rows;
        int block_length = 1;
        int stride = NCOLS; 

        MPI_Type_vector(blockcount, block_length, stride, PersonType, &datatype_column);       // datatype for columns


        int dimensions_full_array[2] = {NROWS, NCOLS};
        int dimensions_sub_array[2] = {r_rows, r_cols};
        int start_coordinates[2] = {0,0};

        MPI_Type_create_subarray(2,  dimensions_full_array, dimensions_sub_array, start_coordinates, MPI_ORDER_C, PersonType, &subMatrix);    // datatype to send the subMatrix


        MPI_Type_commit(&datatype_row);

        MPI_Type_commit(&datatype_column);

        MPI_Type_commit(&subMatrix);


        if ( rank == 0 ){


            MPI_Type_contiguous((r_rows*r_cols), PersonType, &Matrix);     // datatype to receive the subMatrix on the Matrix

            MPI_Type_commit(&Matrix);


            /*al_init();
            al_init_primitives_addon();

            ALLEGRO_DISPLAY *display;

            display = al_create_display((r_cols * dims[1]) * cellSize + 20, (r_rows * dims[0]) * cellSize + 20);
            
            if ( display == NULL ){
                printf("Display error");
            }
            al_set_window_title(display, "DISEASE SPREAD");*/


            readM = new Person[NROWS*NCOLS];
            writeM = new Person[NROWS*NCOLS];
            Mat = new Person[(r_rows*r_cols)*processes];

            SequentialInit();

            for(int s = 0; s < nsteps; s++){

                /*if (s == nsteps - 1){
                    readM[v(1,1)].status = (1000 + rank);
                    readM[v(r_rows, r_cols)].status = (1000 + rank);

                }*/

                BDwork();
                //drawInterface();


                /*if (s == nsteps - 1){
                    print(s);
                }*/

                swap(readM,writeM);

            }

            /*al_destroy_display(display);
            al_uninstall_system();*/




        }

        else{

            readM = new Person[NROWS*NCOLS];
            writeM = new Person[NROWS*NCOLS];

            SequentialInit();

            for(int s = 0; s < nsteps; s++){

                /*if (s == nsteps - 1){
                    readM[v(1,1)].status = (1000 + rank);
                    readM[v(r_rows, r_cols)].status = (1000 + rank);

                }*/

                BDwork();

                swap(readM,writeM);
            }

        }

    }




    double endtime = MPI_Wtime();
    if (rank == 0){
        printf("Elapsed time: %f\n", 1000*(endtime - starttime));
    }

    
    delete[] readM;
    delete[] writeM;
    delete[] Mat;

    // WE FREE ALL DATATYPES

    if ( processes > 1 && check == false){  

        MPI_Type_free(&PersonType);   

        MPI_Type_free(&contiguousRow);

    }

    if (processes > 1 && check == true){

        MPI_Type_free(&PersonType);   

        MPI_Type_free(&datatype_row);
        MPI_Type_free(&datatype_column);
        MPI_Type_free(&subMatrix);

        if ( rank == 0){

            MPI_Type_free(&Matrix);

        }

    }

    MPI_Finalize();
    return 0;

}








void init(){
    // Generating a different seed for every process.
    time_t t1;
    t1 = time(NULL);

    unsigned t2 = (unsigned) t1;
    unsigned v1 = 394671;
    unsigned v2 = 746512;

    t2 = t2 * v1 + v2 + rank;
    
    unsigned v3 = t2 / (rank + 1);
    srand(v3);

    srand(rand() % v3 + rank);

    if (rank != processes -1){
        for(int i=0;i<NROWS/processes+2;i++){
            for(int j=0;j<NCOLS;j++){
                if ( (rand() % 7) == 1 ){
                    readM[v(i,j)].status = rand() % cInf;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
                else{
                    readM[v(i,j)].status = 0;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
            }
        }
    }
    else{
        for(int i=0;i<((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes )))+2;i++){
            for(int j=0;j<NCOLS;j++){
                if ( (rand() % 7) == 1 ){
                    readM[v(i,j)].status = rand() % cInf;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
                else{
                    readM[v(i,j)].status = 0;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
            }
        }
    }   
}

void SequentialInit(){

    time_t t1;
    t1 = time(NULL);

    unsigned t2 = (unsigned) t1;
    unsigned v1 = 394671;
    unsigned v2 = 746512;

    t2 = t2 * v1 + v2 + rank;
    
    unsigned v3 = t2 / (rank + 1);
    srand(v3);

    srand(rand() % v3 + rank);


    if ( check == false ){
        for(int i=0;i<NROWS;i++){
            for(int j=0;j<NCOLS;j++){  
                if ( (rand() % 7) == 1 ){
                    readM[v(i,j)].status = rand() % cInf;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
                else{
                    readM[v(i,j)].status = 0;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
            }
        }
    }

    else{
        for(int i=1;i<NROWS-1;i++){
            for(int j=1;j<NCOLS-1;j++){  
                if ( (rand() % 7) == 1 ){
                    readM[v(i,j)].status = rand() % cInf;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
                else{
                    readM[v(i,j)].status = 0;
                    readM[v(i,j)].T = 0;
                    int r = rand() % 7;
                    if ( r == 2 || r == 3 )
                        readM[v(i,j)].Im = true;
                    else
                        readM[v(i,j)].Im = false;
                }
            }
        }
    }
}




inline void Initial(){
    for(int i=1; i < NROWS/processes +1; i++){
        for(int j=0; j < NCOLS; j++){
            Mat[v(i-1,j)] = readM[v(i,j)];
        }
    }
}




void print(int &step){ 
    printf("Sono il processo %d e printo!\n", rank);
    printf("---%d\n",step);
    if (processes > 1){
        for(int i=0;i<r_rows*dims[0]; i++){
            for(int j=0;j<r_cols*dims[1];j++){
                int p = 0;
                if (Mat[m(i,j)].status != 0)
                    p = Mat[m(i,j)].status;
                printf("%d ",p);
            }
            printf("\n");
        }
    }

    else{
        for(int i=0;i<NROWS;i++){
            for(int j=0;j<NCOLS;j++){
                int p = 0;
                if (readM[v(i,j)].status != 0)
                    p = readM[v(i,j)].status;
                printf("%d ",p);
            }
            printf("\n");
        }
    }
}

void swap(Person * &readM, Person * &writeM){
    Person * p=readM;
    readM=writeM;
    writeM=p;
}



void drawInterface(){
    ALLEGRO_COLOR color;
    if (processes == 1){
        for ( int i = 0; i < NROWS; i++){
            for( int j = 0; j < NCOLS; j++){
                if ((readM[v(i,j)].status == 0) && (readM[v(i,j)].T < soglia)){
                    color = al_map_rgb(51, 0, 0);  // dark red
                }
                else if ((readM[v(i,j)].status == 0) && (readM[v(i,j)].T == soglia)){
                    color = al_map_rgb(138, 43, 226);  // blue purple
                }
                else if (readM[v(i,j)].status > 0 && readM[v(i,j)].status <= 16){
                    color = al_map_rgb(204, 0, 0);  // red
                }
                else if (readM[v(i,j)].status > 16 && readM[v(i,j)].status <= 32){
                    color = al_map_rgb(255, 128, 0);  // orange
                }
                else if (readM[v(i,j)].status > 32 && readM[v(i,j)].status <= 48){
                    color = al_map_rgb(255, 255, 51); // yellow
                }
                else if (readM[v(i,j)].status > 48 && readM[v(i,j)].status <= 64){
                    color = al_map_rgb(0, 153, 76); // green
                }
                else if (readM[v(i,j)].status > 64 && readM[v(i,j)].status <= 80){
                    color = al_map_rgb(255, 255, 255); // white
                }
                else if (readM[v(i,j)].status > 80 && readM[v(i,j)].status <= 99){
                    color = al_map_rgb(128, 128, 128); // gray
                }
                else if (readM[v(i,j)].status == 100){
                    color = al_map_rgb(64, 64, 64); // black
                }
                al_draw_filled_rectangle(i * cellSize + 10, j * cellSize + 10, i * cellSize + 10 + cellSize, j * cellSize + 10 + cellSize, color);
            }
        }
    }

    else if (processes > 1 && check == false){
        for ( int i = 0; i < NROWS; i++){
            for( int j = 0; j < NCOLS; j++){
                if ((Mat[v(i,j)].status == 0) && (Mat[v(i,j)].T < soglia)){
                    color = al_map_rgb(51, 0, 0);  // dark red
                }
                else if ((Mat[v(i,j)].status == 0) && (Mat[v(i,j)].T == soglia)){
                    color = al_map_rgb(138, 43, 226);  // blue purple
                }
                else if (Mat[v(i,j)].status > 0 && Mat[v(i,j)].status <= 16){
                    color = al_map_rgb(204, 0, 0);  // red
                }
                else if (Mat[v(i,j)].status > 16 && Mat[v(i,j)].status <= 32){
                    color = al_map_rgb(255, 128, 0);  // orange
                }
                else if (Mat[v(i,j)].status > 32 && Mat[v(i,j)].status <= 48){
                    color = al_map_rgb(255, 255, 51); // yellow
                }
                else if (Mat[v(i,j)].status > 48 && Mat[v(i,j)].status <= 64){
                    color = al_map_rgb(0, 153, 76); // green
                }
                else if (Mat[v(i,j)].status > 64 && Mat[v(i,j)].status <= 80){
                    color = al_map_rgb(255, 255, 255); // white
                }
                else if (Mat[v(i,j)].status > 80 && Mat[v(i,j)].status <= 99){
                    color = al_map_rgb(128, 128, 128); // gray
                }
                else if (Mat[v(i,j)].status == 100){
                    color = al_map_rgb(64, 64, 64); // black 64
                }
                al_draw_filled_rectangle(i * cellSize + 10, j * cellSize + 10, i * cellSize + 10 + cellSize, j * cellSize + 10 + cellSize, color);
            }
        }
    }

    al_flip_display();  
}


// LA NUOVA FUNZIONE DI TRANSIZIONE !
inline void transFuncCell(int i, int j){
    
    int v;
    int val;

    int infetti = 0;
    int malati = 0;

    int sommaStati = readM[v(i,j)].status;

    if((readM[v(i,j)].status == 0) && (readM[v(i,j)].T < soglia)){
        for(int di=-1;di<2;di++){
            for(int dj=-1;dj<2;dj++){
                if (di!=0 || dj!=0){
                    v = readM[v((i+di+NROWS)%NROWS,(j+dj+NCOLS)%NCOLS)].status;
                    if ( (v > 0) && (v < idM) )
                        infetti++;
                    else if ( v == idM )
                        malati++;
                }
            }
        }

        writeM[v(i,j)].status = ( (infetti / k1) + (malati / k2) );
        writeM[v(i,j)].T = readM[v(i,j)].T;
        writeM[v(i,j)].Im = readM[v(i,j)].Im;
    }



    else if((readM[v(i,j)].status == 0) && (readM[v(i,j)].T == soglia)){
        writeM[v(i,j)].status = readM[v(i,j)].status;
        writeM[v(i,j)].T = readM[v(i,j)].T;
        writeM[v(i,j)].Im = readM[v(i,j)].Im;
    }




    else if((readM[v(i,j)].status > 0) && (readM[v(i,j)].status < idM)){
        for(int di=-1;di<2;di++){
            for(int dj=-1;dj<2;dj++){
                if (di!=0 || dj!=0){
                    v = readM[v((i+di+NROWS)%NROWS,(j+dj+NCOLS)%NCOLS)].status;
                    sommaStati += v;
                    if ( (v > 0) && (v < idM) )
                        infetti++;
                    else if ( v == idM )
                        malati++;
                }
            }
        }

        val = ((sommaStati/(infetti + malati + 1 )) + cInf);
        
        if ( val > idM )
            val = idM;
        
        if ( val < 0 )
            val = 0;

        writeM[v(i,j)].status = val;
        writeM[v(i,j)].T = readM[v(i,j)].T;
        writeM[v(i,j)].Im = readM[v(i,j)].Im;
    }




    else if(readM[v(i,j)].status == idM){

        writeM[v(i,j)].status = 0;

        if (readM[v(i,j)].Im == true)
            writeM[v(i,j)].T = readM[v(i,j)].T + 1;

        else{
            writeM[v(i,j)].T = readM[v(i,j)].T;
            writeM[v(i,j)].Im = readM[v(i,j)].Im;

        }  
    }
}



inline void transFuncBorder(){
    for(int j=0;j<NCOLS;j++){
        transFuncCell(1, j);
        if ( rank != processes -1){
            transFuncCell(NROWS/processes, j);
        }
        else{
            transFuncCell(((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes ))),j);
        }
    }
}

inline void transFuncInside(){
    if ( rank != processes -1){
        for(int i=2;i<NROWS/processes;i++){
            for(int j=0;j<NCOLS;j++){
                transFuncCell(i,j);
            }
        }
    }
    else{
        for(int i=2;i<((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes )));i++){
            for(int j=0;j<NCOLS;j++){
                transFuncCell(i,j);
            }
        }
    }
    
}

inline void transFunc(){
    for(int i = 0 ; i<NROWS ;i++){
        for(int j = 0 ; j<NCOLS ;j++){
            transFuncCell(i,j);
        }
    }
}

inline void receivePropAndWork(int &s){

    MPI_Request r1;
    MPI_Request r2;

    MPI_Isend(&readM[v(NROWS/processes,0)], 1, contiguousRow, down, 17, New_Comm, &r1);    // I START SENDING MY ROWS
    MPI_Isend(&readM[v(1,0)], 1, contiguousRow, up, 20, New_Comm, &r2);

    MPI_Request requests[processes-1];

    for(int p=1; p<processes; p++){     // I START RECEIVING EACH SUBMATRIX IN ORDER
        if (p != processes - 1){
            MPI_Irecv(&Mat[v(((NROWS/processes)*(p)),0)], ((NROWS/processes)*NCOLS), PersonType, p, s + 21, New_Comm, &requests[p-1]);
        }
        else{
            MPI_Irecv(&Mat[v(((NROWS/processes)*(p)),0)], (((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes )))*NCOLS), PersonType, p, s + 21, New_Comm, &requests[p-1]);
        }
    }

    Initial();                         // I SET UP THE MATRIX WITH MY SUBMATRIX
    transFuncInside();                 

    MPI_Status status;                 // I RECEIVE THE ROWS I NEED

    MPI_Recv(&readM[v(0,0)], 1, contiguousRow, up, 17, New_Comm, &status);
    MPI_Recv(&readM[v(NROWS/processes+1,0)], 1, contiguousRow, down, 20, New_Comm, &status);

    transFuncBorder();

    MPI_Wait(&r1 ,MPI_STATUS_IGNORE);  // I ASSURE MYSELF THE ISEND AND IRECV OPERATIONS ARE COMPLETE
    MPI_Wait(&r2 ,MPI_STATUS_IGNORE);

    MPI_Waitall(processes-1, requests, MPI_STATUSES_IGNORE);

}

inline void sendPropAndWork(int &s){

    MPI_Request r1;
    MPI_Request r2;                // I START SENDING MY ROWS

    if ( rank != processes - 1 ){ 
        MPI_Isend(&readM[v(NROWS/processes,0)], 1, contiguousRow, down, 17, New_Comm, &r1);
        MPI_Isend(&readM[v(1,0)], 1, contiguousRow, up, 20, New_Comm, &r2);
    }

    else{
        MPI_Isend(&readM[v(((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes ))),0)], 1, contiguousRow, down, 17, New_Comm, &r1);
        MPI_Isend(&readM[v(1,0)], 1, contiguousRow, up, 20, New_Comm, &r2); 
    }

    
    MPI_Request request;            // I SEND MY SUBMATRIX
    

    if (rank != processes-1){
        MPI_Isend(&readM[v(1,0)], ((NROWS/processes)*NCOLS) , PersonType, 0, s + 21, New_Comm, &request);
    }
    else{
        MPI_Isend(&readM[v(1,0)], (((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes )))*NCOLS) , PersonType, 0, s + 21, New_Comm, &request);
    }


    transFuncInside();


    MPI_Status status;              // I RECEIVE THE ROWS I NEED

    if (rank != processes - 1){
        MPI_Recv(&readM[v(0,0)], 1, contiguousRow, up, 17, New_Comm, &status);
        MPI_Recv(&readM[v(NROWS/processes+1,0)], 1, contiguousRow, down, 20, New_Comm, &status);
    }

    else{
        MPI_Recv(&readM[v(0,0)], 1, contiguousRow, up, 17, New_Comm, &status);
        MPI_Recv(&readM[v(((NROWS/processes)+((NROWS) - ((NROWS/processes) * processes )))+1,0)], 1, contiguousRow, down, 20, New_Comm, &status);
    }


    transFuncBorder();



    MPI_Wait(&r1 ,MPI_STATUS_IGNORE);  // I ASSURE MYSELF THE ISEND OPERATIONS ARE COMPLETE
    MPI_Wait(&r2 ,MPI_STATUS_IGNORE);

    MPI_Wait(&request, MPI_STATUS_IGNORE);
}





void BDwork(){

    // SENDING ALL

    MPI_Request r1, r2, c1, c2, corner1, corner2, corner3, corner4;


    MPI_Isend(&readM[v(1,1)], 1, datatype_row, MAP[0][1], 1, New_Comm, &r1);      // sends the top row to the matrix on top

    MPI_Isend(&readM[v(r_rows, 1)], 1, datatype_row, MAP[2][1], 2, New_Comm, &r2);     // sends the bottom row to the matrix at the bottom

    MPI_Isend(&readM[v(1,1)], 1, datatype_column, MAP[1][0], 3, New_Comm, &c1);      // sends the left column to the matrix on the left

    MPI_Isend(&readM[v(1, r_cols)], 1, datatype_column, MAP[1][2], 4, New_Comm, &c2);     // sends the right column to the matrix on the right

    MPI_Isend(&readM[v(1,1)], 1, PersonType, MAP[0][0], 5, New_Comm, &corner1);      // sends the top left corner to the matrix on the top left

    MPI_Isend(&readM[v(1, r_cols)], 1, PersonType, MAP[0][2], 6, New_Comm, &corner2);     // sends the top right corner to the matrix on the top right

    MPI_Isend(&readM[v(r_rows, 1)], 1, PersonType, MAP[2][0], 7, New_Comm, &corner3);      // sends the bottom left corner to the matrix on the bottom left

    MPI_Isend(&readM[v(r_rows, r_cols)], 1, PersonType, MAP[2][2], 8, New_Comm, &corner4);     // sends the bottom right corner to the matrix on the bottom right

    // _____________________________________________________________________________________
    // IGATHER OPERATION


    MPI_Request request;

    MPI_Igather(&readM[v(1,1)], 1, subMatrix, Mat, 1, Matrix, 0, New_Comm, &request);


    // I WORK EXCLUDING THE BORDERS

    BDtransFuncNoBorder();      

    // I RECEIVE WHAT I NEED 

    MPI_Status status;


    MPI_Recv(&readM[v(r_rows+1, 1)], 1, datatype_row, MAP[2][1], 1, New_Comm, &status);

    MPI_Recv(&readM[v(0,1)], 1, datatype_row, MAP[0][1], 2, New_Comm, &status);

    MPI_Recv(&readM[v(1, r_cols+1)], 1, datatype_column, MAP[1][2], 3, New_Comm, &status);

    MPI_Recv(&readM[v(1,0)], 1, datatype_column, MAP[1][0], 4, New_Comm, &status);

    MPI_Recv(&readM[v(r_rows+1, r_cols+1)], 1, PersonType, MAP[2][2], 5, New_Comm, &status);

    MPI_Recv(&readM[v(r_rows+1, 0)], 1, PersonType, MAP[2][0], 6, New_Comm, &status);

    MPI_Recv(&readM[v(0, r_cols+1)], 1, PersonType, MAP[0][2], 7, New_Comm, &status);

    MPI_Recv(&readM[v(0,0)], 1, PersonType, MAP[0][0], 8, New_Comm, &status);



    // NOW THAT I HAVE ALL, I WORK ON THE BORDERS.

    BDtransFuncOnBorder();



    // I ASSURE MYSELF ALL NON-BLOCKING OPERATIONS ARE COMPLETE.


    MPI_Wait(&r1 ,MPI_STATUS_IGNORE);
    MPI_Wait(&r2 ,MPI_STATUS_IGNORE);
    MPI_Wait(&c1 ,MPI_STATUS_IGNORE);
    MPI_Wait(&c2 ,MPI_STATUS_IGNORE);
    MPI_Wait(&corner1 ,MPI_STATUS_IGNORE);
    MPI_Wait(&corner2 ,MPI_STATUS_IGNORE);
    MPI_Wait(&corner3 ,MPI_STATUS_IGNORE);
    MPI_Wait(&corner4 ,MPI_STATUS_IGNORE);


    MPI_Wait(&request, MPI_STATUS_IGNORE);

}


void BDtransFuncNoBorder(){

    for(int i=2;i < NROWS - 2 ; i++){
            for(int j=2;j < NCOLS - 2 ; j++){
                transFuncCell(i,j);
            }
    }
}

void BDtransFuncOnBorder(){

    for(int j=1; j < NCOLS - 1 ; j++){
        transFuncCell(1, j);
        transFuncCell(r_rows, j);
    }

    for(int k = 2; k < NROWS - 2   ;k++){
        transFuncCell(k, 1);
        transFuncCell(k, r_cols);
    }
}