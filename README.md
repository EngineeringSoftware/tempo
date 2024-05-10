# Tempo

Programming and execution models for parallel bounded exhaustive testing.


## About

Bounded-exhaustive testing (BET), which exercises a program under test
for all inputs up to some bounds, is an effective method for detecting
software bugs. Systematic property-based testing is a BET approach
where developers write test generation programs that describe
properties of test inputs. Hybrid test generation programs offer the
most expressive way to write desired properties by freely combining
declarative filters and imperative generators. However, exploring
hybrid test generation programs, to obtain test inputs, is both
computationally demanding and challenging to parallelize. Tempo is a
framework for parallel exploration of hybrid test generation programs.


## Example

In the example below, we define a CUDA kernel that executes all
possible sequences (up to `n` calls) of `put` and `remove` on
`int_avl_tree_map`.

Let's say that you set `n=1`.  In that case, the kernel below will
execute the following method sequences: (1) `tm.put(0); tm.put(0)`,
(2) `tm.put(0); tm.remove(0)`, (3) `tm.remove(0); tm.put(0)`, (4)
`tm.remove(0); tm.remove(0)`.

```cuda
__global__ void avltSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    Node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_avl_tree_map tm(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            Object* const new_object = env.objectAlloc();
            new_object->id = value;
            tm.put(value, new_object);
        } else {
            tm.remove(value);
        }
    }

    // use input (generated above) to test something
    // e.g., tm.put(tid, new_object);
```

Unlike "normal" CUDA kernels, kernels for Tempo are (automatically)
executed a number of times to explore aforementioned sequences.
Namely, execution of the `_choice` function, stops a tread that
invoked the function and queues tasks for the next kernel invocation;
there will be one new task for every value in the range given by
`_choice`.


## Citation

If you have used Tempo in a research project, please cite this
research
[paper](https://users.ece.utexas.edu/~gligoric/papers/AlAwarETAL21Tempo.pdf):

```bib
@inproceedings{AlAwarETAL21Tempo,
  author = {Al Awar, Nader and Jain, Kush and Rossbach, Christopher J. and Gligoric, Milos},
  title = {Programming and Execution Models for Parallel Bounded Exhaustive Testing},
  booktitle = {Conference on Object-Oriented Programming, Systems, Languages, and Applications},
  pages = {1--28},
  year = {2021},
}
```
