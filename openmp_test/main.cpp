#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>

using namespace std;

void test(int n)
{
	// unordered_set<int> myset;
	int count = 0;
	for (int i = 0; i < n; ++i)
	{
		count += 1.1 * 2.4;
		// myset.insert(i);
	}
	cout << "Sum: " << count << endl;

}

void testParallel(int n)
{
	// unordered_set<int> myset;
	int count = 0;
	int num = n;

#pragma omp parallel for reduction(+: count)
	for (int i = 0; i < num; ++i)
	{
		count += 1.1 * 2.4;
	}

	cout << "Sum: " << count << endl;

	// std::cout << myset.size() << std::endl;
}

int main(int argc, char** argv)
{
	int n = 100000;
	if (argc == 2)
		n = atoi(argv[1]);
	else if (n > 2)
	{
		cout << "Usage: openmp_test [one_large_number (default 100000)]" << endl;
		return -1;
	}

	auto start = std::chrono::steady_clock::now();
	test(n);
	auto end = std::chrono::steady_clock::now();
	double delta = std::chrono::duration_cast<chrono::seconds>(end - start).count();
	cout << "Time: " + std::to_string(delta) << endl;

	start = std::chrono::steady_clock::now();
	testParallel(n);
	end = std::chrono::steady_clock::now();
	delta = std::chrono::duration_cast<chrono::seconds>(end - start).count();
	cout << "Time (Parallel): " + std::to_string(delta) << endl;

	return 0;
}