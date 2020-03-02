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
