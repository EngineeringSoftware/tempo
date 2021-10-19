#include <stdio.h>

#include "taco_avlt.h"
#include "../consts.h"

avl_node* Env::nodeAlloc() {
    avl_node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

avl_tree::avl_tree(Env *env)
    : root(nullptr), env(env) {
}

avl_node* avl_tree::doubleWithLeftChild(avl_node* k3) {
    k3->left = avl_tree::rotateWithRightChild(k3->left);
    return avl_tree::rotateWithLeftChild(k3);
}

avl_node* avl_tree::doubleWithRightChild(avl_node* k1) {
    k1->right = avl_tree::rotateWithLeftChild(k1->right);
    return avl_tree::rotateWithRightChild(k1);
}

int avl_tree::height(avl_node* t) {
    return t == nullptr ? -1 : t->height;
}

int avl_tree::max(int lhs, int rhs) {
    return lhs > rhs ? lhs : rhs;
}

avl_node* avl_tree::rotateWithLeftChild(avl_node* k2) {
    avl_node* k1 = k2->left;
    k2->left = k1->right;
    k1->right = k2;
    k2->height = avl_tree::max(avl_tree::height(k2->left), avl_tree::height(k2->right)) + 1;
    k1->height = avl_tree::max(avl_tree::height(k1->left), k2->height) + 1;
    return k1;
}

avl_node* avl_tree::rotateWithRightChild(avl_node* k1) {
    avl_node* k2 = k1->right;
    k1->right = k2->left;
    k2->left = k1;
    k1->height = avl_tree::max(avl_tree::height(k1->left), avl_tree::height(k1->right)) + 1;
    k2->height = avl_tree::max(avl_tree::height(k2->right), k1->height) + 1;
    return k2;
}

bool avl_tree::balanced() {
    return balanced(this->root);
}

bool avl_tree::balanced(avl_node* an) {
    if (an == nullptr) {
        return true;
    }

    const int lh = avl_tree::height(an->left);
    const int rh = avl_tree::height(an->right);

    return ((lh == rh) || (lh == rh + 1) || (lh + 1 == rh))
        && balanced(an->left) && balanced(an->right);
}

int avl_tree::elementAt(avl_node* t) {
    return t == nullptr ? -1 : t->element;
}

int avl_tree::find(int x) {
    return elementAt(find(x, this->root));
}

avl_node* avl_tree::findNode(int x) {
    return find(x, this->root);
}

avl_node* avl_tree::find(int x, avl_node* arg) {
    avl_node* t = arg;
    while (t != nullptr) {
        if (x < t->element) {
            t = t->left;
        } else {
            if (x > t->element) {
                t = t->right;
            } else {
                return t;
            }
        }
    }

    return nullptr;
}

int avl_tree::findMax() {
    return elementAt(findMax(this->root));
}

avl_node* avl_tree::fmax() {
    return findMax(this->root);
}

avl_node* avl_tree::findMax(avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        return t;
    }

    while (t->right != nullptr) {
        t = t->right;
    }

    return t;
}

int avl_tree::findMin() {
    return elementAt(findMin(this->root));
}

avl_node* avl_tree::findMin(avl_node* t) {
    if (t == nullptr) {
        return t;
    }

    while (t->left != nullptr) {
        t = t->left;
    }

    return t;
}

void avl_tree::insertElem(int x) {
    this->root = insert(x, this->root);
}

avl_node* avl_tree::insert(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            t->left = insert(x, t->left);
            if (t->left == nullptr) {
                return nullptr;
            }
            
            if (avl_tree::height(t->left) - avl_tree::height(t->right) == 2) {
                if (x < t->left->element) {
                    t = avl_tree::rotateWithLeftChild(t);
                } else {
                    t = avl_tree::doubleWithLeftChild(t);
                }
            }
        } else {
            if (x > t->element) {
                t->right = insert(x, t->right);
                if (t->right == nullptr) {
                    return nullptr;
                }

                if (avl_tree::height(t->right) - avl_tree::height(t->left) == 2) {
                    if (x > t->right->element) {
                        t = avl_tree::rotateWithRightChild(t);
                    } else {
                        t = avl_tree::doubleWithRightChild(t);
                    }
                }
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), avl_tree::height(t->right)) + 1;

    return t;
}

avl_node* avl_tree::insert_0(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            t->left = insert_1(x, t->left);
            if (t->left == nullptr) {
                return nullptr;
            }
            
            if (avl_tree::height(t->left) - avl_tree::height(t->right) == 2) {
                if (x < t->left->element) {
                    t = avl_tree::rotateWithLeftChild(t);
                } else {
                    t = avl_tree::doubleWithLeftChild(t);
                }
            }
        } else {
            if (x > t->element) {
                t->right = insert_1(x, t->right);
                if (t->right == nullptr) {
                    return nullptr;
                }

                if (avl_tree::height(t->right) - avl_tree::height(t->left) == 2) {
                    if (x > t->right->element) {
                        t = avl_tree::rotateWithRightChild(t);
                    } else {
                        t = avl_tree::doubleWithRightChild(t);
                    }
                }
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), avl_tree::height(t->right)) + 1;

    return t;
}

avl_node* avl_tree::insert_1(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            t->left = insert_2(x, t->left);
            if (t->left == nullptr) {
                return nullptr;
            }
            
            if (avl_tree::height(t->left) - avl_tree::height(t->right) == 2) {
                if (x < t->left->element) {
                    t = avl_tree::rotateWithLeftChild(t);
                } else {
                    t = avl_tree::doubleWithLeftChild(t);
                }
            }
        } else {
            if (x > t->element) {
                t->right = insert_2(x, t->right);
                if (t->right == nullptr) {
                    return nullptr;
                }

                if (avl_tree::height(t->right) - avl_tree::height(t->left) == 2) {
                    if (x > t->right->element) {
                        t = avl_tree::rotateWithRightChild(t);
                    } else {
                        t = avl_tree::doubleWithRightChild(t);
                    }
                }
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), avl_tree::height(t->right)) + 1;

    return t;
}

avl_node* avl_tree::insert_2(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            t->left = insert_3(x, t->left);
            if (t->left == nullptr) {
                return nullptr;
            }
            
            if (avl_tree::height(t->left) - avl_tree::height(t->right) == 2) {
                if (x < t->left->element) {
                    t = avl_tree::rotateWithLeftChild(t);
                } else {
                    t = avl_tree::doubleWithLeftChild(t);
                }
            }
        } else {
            if (x > t->element) {
                t->right = insert_3(x, t->right);
                if (t->right == nullptr) {
                    return nullptr;
                }

                if (avl_tree::height(t->right) - avl_tree::height(t->left) == 2) {
                    if (x > t->right->element) {
                        t = avl_tree::rotateWithRightChild(t);
                    } else {
                        t = avl_tree::doubleWithRightChild(t);
                    }
                }
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), avl_tree::height(t->right)) + 1;

    return t;
}

avl_node* avl_tree::insert_3(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            t->left = insert_4(x, t->left);
            if (t->left == nullptr) {
                return nullptr;
            }

            if (avl_tree::height(t->left) - avl_tree::height(t->right) == 2) {
                if (x < t->left->element) {
                    t = avl_tree::rotateWithLeftChild(t);
                } else {
                    t = avl_tree::doubleWithLeftChild(t);
                }
            }
        } else {
            if (x > t->element) {
                t->right = insert_4(x, t->right);
                if (t->right == nullptr) {
                    return nullptr;
                }

                if (avl_tree::height(t->right) - avl_tree::height(t->left) == 2) {
                    if (x > t->right->element) {
                        t = avl_tree::rotateWithRightChild(t);
                    } else {
                        t = avl_tree::doubleWithRightChild(t);
                    }
                }
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), avl_tree::height(t->right)) + 1;

    return t;
}

avl_node* avl_tree::insert_4(int x, avl_node* arg) {
    avl_node* t = arg;
    if (t == nullptr) {
        t = env->nodeAlloc();
        t->element = x;
        t->left = nullptr;
        t->right = nullptr;
    } else {
        if (x < t->element) {
            return nullptr;
        } else {
            if (x > t->element) {
                return nullptr;
            }
        }
    }

    t->height = avl_tree::max(avl_tree::height(t->left), - avl_tree::height(t->right)) + 1;

    return t;
}

bool avl_tree::isEmpty() {
    return this->root == nullptr;
}

void avl_tree::makeEmpty() {
    delete this->root;
    this->root = nullptr;
}

bool avl_tree::maxElement(int max) {
    return maxElement(max, this->root);
}

bool avl_tree::maxElement(int max, avl_node* t) {
    if (t == nullptr) {
        return true;
    }

    if (max < t->element) {
        return false;
    }

    return maxElement(max, t->left) && maxElement(max, t->right);
}

bool avl_tree::minElement(int min) {
    return minElement(min, this->root);
}

bool avl_tree::minElement(int min, avl_node* t) {
    if (t == nullptr) {
        return true;
    }

    if (min > t->element) {
        return false;
    }

    return minElement(min, t->left) && minElement(min, t->right);
}

void avl_tree::remove(int x) {
}

bool avl_tree::wellFormed() {
    return wellFormed(this->root);
}

int avl_tree::mathMax(int l, int r) {
    return (l > r ? l : r);
}

bool avl_tree::wellFormed(avl_node* an) {
    if (an == nullptr) {
        return true;
    }

    if (avl_tree::height(an) != mathMax(avl_tree::height(an->left), avl_tree::height(an->right)) + 1) {
        return false;
    }

    return wellFormed(an->left) && wellFormed(an->right);
}
