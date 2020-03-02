#include <iostream>

using namespace std;

int readNumber();
void writeAnswer(int);

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
