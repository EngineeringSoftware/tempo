
class Record {
public:
    int parent;
    int rank;

    __host__ __device__ Record() {}
    __host__ __device__ Record(const Record& rec) : parent(rec.parent), rank(rec.rank) {}
};

class disjset {
private:
    Record* elements;
    int size;
    bool PATH_COMPRESSION;
    // int findForToString(int el);

public:
    __host__ __device__ disjset(Record* elements);
    __host__ __device__ void setPathCompression(bool value);
    __host__ __device__ void create(int n);
    __host__ __device__ int find(int x);
    __host__ __device__ void unionMethod(int x, int y);
    // char* toString();
    // __host__ __device__ bool allDifferent();
    // methods used by Korat
    __host__ __device__ bool repOK();
};