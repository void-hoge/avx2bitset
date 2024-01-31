test: avx2bitset.hpp test.cpp
	g++ test.cpp -march=native -Wall -Wextra -o test

clean:
	rm test
