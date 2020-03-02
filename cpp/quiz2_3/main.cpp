#include <iostream>
#include "io.h"

using namespace std;

int main(int argc, char const *argv[])
{
    int firstNumber{};
    int secondNumber{};
    cout << "First number: ";
    firstNumber = readNumber();
    cout << "Second number: ";
    secondNumber = readNumber();
    writeAnswer(firstNumber + secondNumber);
    return 0;
}

