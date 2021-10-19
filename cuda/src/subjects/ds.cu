#include <stdio.h>

#include "ds.h"
#include "../consts.h"

disjset::disjset(Record* elements) : elements(elements), size(0) {
}

void disjset::setPathCompression(bool value) {
    PATH_COMPRESSION = value;
}

void disjset::create(int n) {
    size = n;
    for(int i = 0; i < n; i++) {
        elements[i].parent = i;
        elements[i].rank = 0;
    }
}

int disjset::find(int x) {
    if (elements[x].parent != x)
        if(PATH_COMPRESSION)
            elements[x].parent = find(elements[x].parent);
        else
            return find(elements[x].parent);
    return elements[x].parent;
}

void disjset::unionMethod(int x, int y) {
    int px = find(x);
    int py = find(y);
    
    /*
    if (px < py) {
      int t = px;
      px = py;
      py = t;
    } */
    
    if (px == py)
      return;
    
    if (elements[px].rank > elements[py].rank)
      elements[py].parent = px;
    else
      elements[px].parent = py;

    if (elements[px].rank == elements[py].rank)
      elements[py].rank++;
}

// bool disjset::allDifferent() {
//     int n = size - 1;
//     // for (int n = 1; n < size; n++) // generates fewer candidates, but
//     // slower
//     for (int i = 0; i < n; i++)
//       for (int j = i + 1; j <= n; j++)
//         if (elements[i] == elements[j])
//           return false;
//     return true;
// }