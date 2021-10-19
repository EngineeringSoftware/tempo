#include "cl.h"
#include "../consts.h"

linked_list_node* Env::nodeAlloc() {
    linked_list_node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

Object* Env::objectAlloc() {
    Object *new_object = &(objectPool[objectPoolIndex]);
    objectPoolIndex++;
    if (objectPoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in objectPool, Index %d\n", objectPoolIndex);
    }

    return new_object;
}

node_caching_linked_list::node_caching_linked_list(Env *env)
    : size(0), modCount(0), DEFAULT_MAXIMUM_CACHE_SIZE(20), header(createHeaderNode()),
    firstCachedNode(nullptr), cacheSize(0), maximumCacheSize(DEFAULT_MAXIMUM_CACHE_SIZE),
    env(env) {
}

linked_list_node* node_caching_linked_list::createHeaderNode() {
    return env->nodeAlloc();
}

int node_caching_linked_list::getMaximumCacheSize() {
    return maximumCacheSize;
}

void node_caching_linked_list::setMaximumCacheSize(int maximumCacheSize) {
    this->maximumCacheSize = maximumCacheSize;
    shrinkCacheToMaximumSize();
}

void node_caching_linked_list::shrinkCacheToMaximumSize() {
    while (cacheSize > maximumCacheSize) {
        getNodeFromCache();
    }
}

linked_list_node* node_caching_linked_list::getNodeFromCache() {
    if (cacheSize == 0) {
        return nullptr;
    }
    linked_list_node* cached_node = firstCachedNode;
    firstCachedNode = cached_node->next;
    cached_node->next = nullptr;
    cacheSize--;

    return cached_node;
}

bool node_caching_linked_list::isCacheFull() {
    return cacheSize >= maximumCacheSize;
}

void node_caching_linked_list::addNodeToCache(linked_list_node* node) {
    if (isCacheFull()) {
        return;
    }
    linked_list_node* next_cached_node = firstCachedNode;
    node->previous = nullptr;
    node->next = next_cached_node;
    node->setValue(nullptr);
    firstCachedNode = node;
    cacheSize++;
}

linked_list_node* node_caching_linked_list::createNode(Object* value) {
    linked_list_node* cached_node = getNodeFromCache();
    if (cached_node == nullptr) {
        return superCreateNode(value);
    } else {
        cached_node->setValue(value);
        return cached_node;
    }
}

void node_caching_linked_list::superRemoveNode(linked_list_node* node) {
    node->previous->next = node->next;
    node->next->previous = node->previous;
    size--;
    modCount++;
}

void node_caching_linked_list::removeNode(linked_list_node* node) {
    superRemoveNode(node);
    addNodeToCache(node);
}

void node_caching_linked_list::removeAllNodes() {
    int number_of_nodes_to_cache = mathMin(size, maximumCacheSize - cacheSize);
    linked_list_node* node = header->next;
    for (int current_index = 0; current_index < number_of_nodes_to_cache; current_index++) {
        linked_list_node* old_node = node;
        node = node->next;
        addNodeToCache(old_node);
    }

    superRemoveAllNodes();
}

int node_caching_linked_list::mathMin(int left, int right) {
    return left < right ? left : right;
}

linked_list_node* node_caching_linked_list::superCreateNode(Object* value) {
    linked_list_node *node = env->nodeAlloc();
    node->value = value;
    return node;
}

void node_caching_linked_list::superRemoveAllNodes() {
    header->next = header;
    header->previous = header;
    size = 0;
    modCount++;
}

bool node_caching_linked_list::remove(const Object& value) {
    for (linked_list_node* node = header->next; node != header; node = node->next) {
        if (isEqualValue(*(node->value), value)) {
            removeNode(node);
            return true;
        }
    }

    return false;
}

bool node_caching_linked_list::isEqualValue(const Object& value_1, const Object& value_2) {
    return value_1.id == value_2.id;
}

bool node_caching_linked_list::add(Object* value) {
    addLast(value);
    return true;
}

bool node_caching_linked_list::addLast(Object* value) {
    addNodeBefore(header, value);
    return true;
}

void node_caching_linked_list::addNodeBefore(linked_list_node* node, Object* value) {
    linked_list_node* new_node = createNode(value);
    addNode(new_node, node);
}

void node_caching_linked_list::addNode(linked_list_node* node_to_insert, linked_list_node* insert_before_node) {
    node_to_insert->next = insert_before_node;
    node_to_insert->previous = insert_before_node->previous;
    insert_before_node->previous->next = node_to_insert;
    insert_before_node->previous = node_to_insert;
    size++;
    modCount++;
}

bool node_caching_linked_list::contains(const Object& value) {
    return indexOf(value) != -1;
}

int node_caching_linked_list::indexOf(const Object& value) {
    int i = 0;
    for (linked_list_node* node = header->next; node != header; node = node->next) {
        if (isEqualValue(*(node->value), value)) {
            return i;
        }
        i++;
    }

    return -1;
}

Object* node_caching_linked_list::removeIndex(int index) {
    linked_list_node* node = getNode(index, false);
    Object* old_value = node->getValue();
    removeNode(node);
    return old_value;
}

linked_list_node* node_caching_linked_list::getNode(int index, bool end_marker_allowed) {
    if (index < 0) {
        return nullptr;
    }
    if (!end_marker_allowed && index == size) {
        return nullptr;
    }
    if (index > size) {
        return nullptr;
    }

    linked_list_node* node;
    if (index < (size / 2)) {
        node = header->next;
        for (int current_index = 0; current_index < index; current_index++) {
            node = node->next;
        }
    } else {
        node = header;
        for (int current_index = size; current_index > index; current_index--) {
            node = node->previous;
        }
    }

    return node;
}
