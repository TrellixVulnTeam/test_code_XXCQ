#include "myheap.h"
#include <unordered_map>

class TestMyHeap
{
public:
    struct Edge : public MxHeapable
    {
        int v0, v1;
        Edge(int a, int b) : v0(a), v1(b) {}
    };

    TestMyHeap() {}
    ~TestMyHeap()
    {
        while (heap_.size() > 0)
        {
            Edge* edge = (Edge*)heap_.extract();
            delete edge;
            edge = nullptr;
        }
    }

    void createHeap()
    {
        Edge* e = new Edge(2, 3);
        e->heap_key(-4);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(0, 1);
        e->heap_key(-0);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(0, 4);
        e->heap_key(-1);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(4, 5);
        e->heap_key(-7);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(1, 2);
        e->heap_key(-2);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(5, 6);
        e->heap_key(-6);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(3, 6);
        e->heap_key(-5);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);

        e = new Edge(0, 3);
        e->heap_key(-3);
        heap_.insert(e);
        all_edges_[e->v0].push_back(e);
        all_edges_[e->v1].push_back(e);
    }

    void run()
    {
        createHeap();
        std::cout << "Heap Size: " << heap_.size() << std::endl;

        int n = heap_.size();
        for (int i = 0; i < n; ++i)
        {
            Edge* e = (Edge*)heap_.extract();
            std::cout << "Edge " << i << ": " << e->v0 << "," << e->v1 << ", Energy: " << e->heap_key() << std::endl;
            delete e;
            e = nullptr;
        }
    }

private:
    MxHeap heap_;
    std::unordered_map<int, std::vector<Edge*>> all_edges_;
};

int main()
{
    TestMyHeap test;
    test.run();
    return 0;
}