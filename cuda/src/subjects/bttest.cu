
#include "bt.h"

#define NUM_INSTANCES 100

int main(void) {
    Node nodes[NUM_INSTANCES] = { 0 };
    Env env(nodes);
    
    BT bt(&env);
    bt.add(4);
    bt.add(2);
    bt.add(6);
    bt.add(5);
    bt.add(7);
    bt.add(1);
    bt.add(3);
    bt.remove(4);
    bt.print();

    return 0;
}
