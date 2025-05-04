#include <iostream>
#include <algorithm>

int hammingDistance(int n1, int n2) {
    int x = n1 ^ n2;
    int setBits = 0;
    while (x > 0) {
        setBits += x & 1;
        x >>= 1;
    }
    return setBits;
}

int main() {
    int num1, num2;
    std::cout << "Enter two integers: ";
    std::cin >> num1 >> num2;
    std::cout << "Hamming Distance: " << hammingDistance(num1, num2) << std::endl;
    return 0;
}


// This code calculates the Hamming distance by performing a bitwise XOR operation on the two input integers. 
//The number of set bits (1s) in the result is then counted, which represents the Hamming distance.


// Hopfield networks distinguish patterns better when they are very different from each other (have a large Hamming distance).
// If patterns share too many common elements, the network can easily get confused during recall and converge to the wrong attractor state.