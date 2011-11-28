/**
*   @Brief: Reonocimiento de rostros, basado en "eigenface.c, by Robin Hewitt, 2007"
*
*
**/

#include <iostream>
#include "Headers/CFace.hpp"    // Clase que implementa la interfaz basica

using namespace std;

int main( int argc, char** argv )
{
    CFace Test;
    Test.Execute_Capture();
    return 0;
}
