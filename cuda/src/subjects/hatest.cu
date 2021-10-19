
#include "ha.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    int array[NUM_INSTANCES] = { 0 };
    HA ha(array);

    ha.insert(4);
    ha.insert(2);
    ha.insert(6);
    ha.insert(5);
    ha.insert(7);
    ha.insert(1);
    ha.insert(3);
    ha.remove();
    ha.print();

    assert(ha.getSize() == 5);
    return 0;
}
