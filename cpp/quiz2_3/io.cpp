#include "io.h"

#include <iostream>

using namespace std;

int readNumber()
{
    int number;
    cin >> number;
    return number;
}

void writeAnswer(int value)
{
    cout << "Answer is " << value << '\n';
}