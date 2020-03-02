#include <iostream>

int main() {
    std::cout << "Please enter a number: ";
    int num = 0;
    std::cin >> num;
    std::cout << "Double that number is: " << num *2 << '\n';
    std::cout << "Triple that number is: " << num *3 << '\n';
    return 0;
}