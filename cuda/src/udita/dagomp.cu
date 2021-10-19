
#include "dag.h"
#include "../ompmains.h"

ImpObjPool(Node);

int8_t nodeProperty(Node *n, LinkedList *path, Set *visited) {
    assert(path->size == 0);

    LinkedList work;
    llInit(&work);
    llAdd(&work, n);

    while (work.size != 0) {
        Node *current = llRemoveLast(&work);
        if (current == NULL) {
            llRemoveLast(path);
            continue;
        }

        // if not acyclic
        if (llContains(path, current)) {
            return FALSE;
        }

        llAdd(path, current);
        llAdd(&work, NULL);
        setAdd(visited, current);

        // check for diamond
        for (int32_t i = 0; i < current->num_of_children; i++) {
            Node *child = current->children[i];
            for (int32_t j = 0; j < i; j++) {
                if (child == current->children[j]) {
                    return FALSE;
                }
            }
        }

        // add all children
        for (int32_t i = 0; i < current->num_of_children; i++) {
            if (current->children[i] != NULL) {
                llAdd(&work, current->children[i]);
            }
        }
    }

    return TRUE;
}

int8_t nodePropertyBuggy(Node *n, Set *path, Set *visited) {
    if (setContains(path, n)) {
        return FALSE;
    }

    setAdd(path, n);
    setAdd(visited, n);
    for (int32_t i = 0; i < n->num_of_children; i++) {
        Node *child = n->children[i];
        // two children of a DAG cannot be the same object
        for (int32_t j = 0; j < i; j++) {
            if (child == n->children[j]) {
                return FALSE;
            }
        }
        // check property for every child of this node
        if (child != NULL && !(nodePropertyBuggy(child, path, visited))) {
            return FALSE;
        }
    }

    setRemove(path, n);
    return TRUE;
}

int8_t dagProperty(Env *env, DAG *dag) {
    LinkedList ll_visited;
    Set visited;
    setInit(&visited, &ll_visited);

    LinkedList path;
    llInit(&path);

    LinkedList work;
    llInit(&work);

    llAdd(&work, dag->root);
    while (work.size != 0) {
        Node *next = llRemoveFirst(&work);
        if (!setContains(&visited, next)) {
            if (!nodeProperty(next, &path, &visited)) {
                return FALSE;
            }
        }
    }
    return (setSize(&visited) == env->num_of_nodes);
}

void dagPrint(Node *root, char graph[]) {
    LinkedList work;
    llInit(&work);
    llAdd(&work, root);
    int index = 0;

    while (work.size != 0) {
        Node *current = llRemoveLast(&work);
        if (current == NULL) {
            graph[index++] = 'N';
            graph[index++] = '-';
            graph[index++] = '[';
            graph[index++] = ']';
            graph[index++] = '-';
            continue;
        } else {
            graph[index++] = current->id + '0';
            graph[index++] = '-';
            if (current->num_of_children == 0) {
                graph[index++] = '[';
                graph[index++] = ']';
                graph[index++] = '-';
            }
        }

        // add all children
        for (int32_t i = 0; i < current->num_of_children; i++) {
            llAdd(&work, current->children[i]);
        }
    }
    graph[index++] = 'e';
    graph[index] = '\0';
    int tid = omp_get_thread_num();

    printf("Graph of id %d: %s\n", tid, graph);
}

void dagGenerate(Env *env, DAG *dag) {
    NodePool *op = env->op;
    dag->root = op->getNew(op);

    for (int32_t i = 0; i < op->size; i++) {
        Node *n = op->getObject(op, i);
        if (n != NULL) {
            n->id = i;
            int32_t num_of_children = _choice(0, env->num_of_nodes - 1);
            n->num_of_children = num_of_children;
            for (int32_t j = 0; j < num_of_children; j++) {
                n->children[j] = op->getAny(op);
            }
        }
    }
}

void dagUdita(int32_t size) {
    int tid = omp_get_thread_num();

    NodePool op;
    initNodePool(&op, size, INCLUDE_NULL);

    Env env = {
        .num_of_nodes = size,
        .op = &op,
    };

    DAG dag = {
        .root = NULL,
    };

    dagGenerate(&env, &dag);
    int8_t is_dag = dagProperty(&env, &dag);
    _countIf(is_dag);
    _ignoreIf(!is_dag);
    // if (is_dag) {
    //     char graph[50];
    //     dagPrint(dag.root, graph);
    // }
}

void testNodeProperty(int32_t bck_active) {
    LinkedList ll_path;
    Set path;
    setInit(&path, &ll_path);

    LinkedList ll_visited;
    Set visited;
    setInit(&visited, &ll_visited);

    Node node;
    nodePropertyBuggy(&node, &path, &visited);
}

int main(int argc, char* argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))dagUdita);
}
