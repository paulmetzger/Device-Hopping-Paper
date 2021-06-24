//
// Created by paul on 15/06/2020.
//

#include <cstdlib>

// ------------------- TODO the following define and structs should be in the common.h header --------------------------------
// ------------------------ but the source to source cl translation needs them in the benchmark ------------------------------

#define  DEFAULT_ORDER 128

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 

typedef struct record {
	int value;
} record;

// ----------------------------------------------------------------------------------------------------------------------------

#include "common.h"


void
list_init(	list_t *l,
              int32_t (*compare)(const void *key,
                                 const void *with),
              void (*datum_delete)(void *))
{
    l->head = l->tail = NULL;
    l->length = 0;
    l->compare = compare;
    l->datum_delete = datum_delete;
}

void
list_delete(list_t *l)
{

    list_item_t *li, *del;

    for (li = l->head; li;) {

        del = li;
        li = li->next;
        list_item_delete(del, l->datum_delete);
    }

    l->head = l->tail = NULL;
    l->length = 0;
}

void
list_reset(list_t *l)
{
    list_delete(l);
}

void
list_insert_item_head(	list_t *l,
                          list_item_t *i)
{
    if (l->head) {
        i->next = l->head;
        l->head->pred = i;
        l->head = i;
        l->head->pred = NULL;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void
list_insert_item_tail(	list_t *l,
                          list_item_t *i)
{
    if (l->head) {
        l->tail->next = i;
        i->pred = l->tail;
        i->next = NULL;
        l->tail = i;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void
list_insert_item_before(list_t *l,
                        list_item_t *next,
                        list_item_t *i)
{
    /* Assume next is actually in the list! */
    /* If it's not, we may lose the list.   */
    if (l->head == next) {
        i->next = next;
        i->pred = NULL;
        l->head = i;
        next->pred = i;
    } else {
        i->next = next;
        i->pred = next->pred;
        next->pred->next = i;
        next->pred = i;
    }
    l->length++;
}

void
list_insert_item_after(	list_t *l,
                           list_item_t *pred,
                           list_item_t *i)
{
    /* Assume pred is actually in the list! */
    /* If it's not, we may lose the list.   */
    if (l->tail == pred) {
        i->pred = pred;
        i->next = NULL;
        l->tail = i;
        pred->next = i;
    } else {
        i->pred = pred;
        i->next = pred->next;
        pred->next->pred = i;
        pred->next = i;
    }
    l->length++;
}

void
list_insert_item_sorted(list_t *l,
                        list_item_t *i)
{
    list_item_t *itr;

    if (l->head) {
        for (itr = l->head; itr && l->compare(list_item_get_datum(i),
                                              list_item_get_datum(itr)) < 0;
             itr = itr->next)
            ;
        if (itr) {
            i->next = itr;
            i->pred = itr->pred;
            itr->pred = i;
            i->pred->next = i;
        } else {
            l->tail->next = i;
            i->pred = l->tail;
            i->next = NULL;
            l->tail = i;
        }
    } else {
        l->head = l->tail = i;
        i->pred = i->next = NULL;
    }
    l->length++;
}

void
list_insert_head(	list_t *l,
                     void *v)
{
    list_item_t *i;
    i = (list_item_t *)malloc(sizeof (*i));
    list_item_init(i, v);
    if (l->head) {
        i->next = l->head;
        l->head->pred = i;
        l->head = i;
        l->head->pred = NULL;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;

}

void
list_insert_tail(	list_t *l,
                     void *v)
{
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof (*i));
    list_item_init(i, v);
    if (l->head) {
        l->tail->next = i;
        i->pred = l->tail;
        i->next = NULL;
        l->tail = i;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void
list_insert_before(	list_t *l,
                       list_item_t *next,
                       void *v)
{
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof (*i));
    list_item_init(i, v);

    /* Assume next is actually in the list! */
    /* If it's not, we may lose the list.   */
    if (l->head == next) {
        i->next = next;
        i->pred = NULL;
        l->head = i;
        next->pred = i;
    } else {
        i->next = next;
        i->pred = next->pred;
        next->pred->next = i;
        next->pred = i;
    }
    l->length++;
}

void
list_insert_after(	list_t *l,
                      list_item_t *pred,
                      void *v)
{
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof (*i));
    list_item_init(i, v);

    /* Assume pred is actually in the list! */
    /* If it's not, we may lose the list.   */
    if (l->tail == pred) {
        i->pred = pred;
        i->next = NULL;
        l->tail = i;
        pred->next = i;
    } else {
        i->pred = pred;
        i->next = pred->next;
        pred->next->pred = i;
        pred->next = i;
    }
    l->length++;
}

void
list_insert_sorted(	list_t *l,
                       void *v)
{
    list_item_t *itr;
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof (*i));
    list_item_init(i, v);


    if (l->head) {
        for (itr = l->head; itr && l->compare(list_item_get_datum(i),
                                              list_item_get_datum(itr)) < 0;
             itr = itr->next)
            ;
        if (itr) {
            i->next = itr;
            i->pred = itr->pred;
            itr->pred = i;
            i->pred->next = i;
        } else {
            l->tail->next = i;
            i->pred = l->tail;
            i->next = NULL;
            l->tail = i;
        }
    } else {
        l->head = l->tail = i;
        i->pred = i->next = NULL;
    }
    l->length++;
}

void
list_remove_item(	list_t *l,
                     list_item_t *i)
{
    if (i == l->head) {
        l->head = l->head->next;
        if (l->head)
            l->head->pred = NULL;
        else
            l->tail = NULL;
    } else if (i == l->tail) {
        l->tail = l->tail->pred;
        l->tail->next = NULL;
    } else {
        i->pred->next = i->next;
        i->next->pred = i->pred;
    }
    l->length--;
    list_item_delete(i, l->datum_delete);
}

void
list_remove_head(list_t *l)
{
    list_remove_item(l, l->head);
}

void
list_remove_tail(list_t *l)
{
    list_remove_item(l, l->tail);
}

list_item_t*
list_find_item(	list_t *l,
                   void *datum)
{
    list_item_t *li;

    for (li = l->head; li && l->compare(datum, list_item_get_datum(li));
         li = li->next)
        ;

    return li;
}

list_item_t*
list_get_head_item(list_t *l)
{
    return l->head;
}

list_item_t*
list_get_tail_item(list_t *l)
{
    return l->tail;
}

void*
list_find(	list_t *l,
              void *datum)
{
    list_item_t *li;

    for (li = l->head; li && l->compare(datum, list_item_get_datum(li));
         li = li->next)
        ;

    return li ? li->datum : NULL;
}

void*
list_get_head(list_t *l)
{
    return l->head ? l->head->datum : NULL;
}

void*
list_get_tail(list_t *l)
{
    return l->tail ? l->tail->datum : NULL;
}

uint32_t
list_get_length(list_t *l)
{
    return l->length;
}

bool
list_is_empty(list_t *l)
{
    return (l->length == 0);
}

bool
list_not_empty(list_t *l)
{
    return (l->length != 0);
}

void
list_visit_items(	list_t *l,
                     void (*visitor)(void *v))
{
    list_item_t *li;

    for (li = l->head; li; li = li->next)
        visitor(list_item_get_datum(li));
}

void
list_item_init(	list_item_t *li,
                   void *datum)
{
    li->pred = li->next = NULL;
    li->datum = datum;
}

void
list_item_delete(	list_item_t *li,
                     void (*datum_delete)(void *datum))
{
    if (datum_delete) {
        datum_delete(li->datum);
    }

    free(li);
}

void *
list_item_get_datum(list_item_t *li)
{
    return li->datum;
}

void
list_iterator_init(	list_t *l,
                       list_iterator_t *li)
{
    *li = l ? l->head : NULL;
}

void
list_iterator_delete(list_iterator_t *li)
{
    *li = NULL;
}

void
list_iterator_next(list_iterator_t *li)
{
    if (*li)
        *li = (*li)->next;
}

void
list_iterator_prev(list_iterator_t *li)
{
    if (*li)
        *li = (*li)->pred;
}

void *
list_iterator_get_datum(list_iterator_t *li)
{
    return *li ? (*li)->datum : NULL;
}

bool
list_iterator_is_valid(list_iterator_t *li)
{
    return (*li != NULL);
}

void
list_reverse_iterator_init(	list_t *l,
                               list_reverse_iterator_t *li)
{
    *li = l ? l->tail : NULL;
}

void
list_reverse_iterator_delete(list_reverse_iterator_t *li)
{
    *li = NULL;
}

void
list_reverse_iterator_next(list_reverse_iterator_t *li)
{
    if (*li)
        *li = (*li)->pred;
}

void
list_reverse_iterator_prev(list_reverse_iterator_t *li)
{
    if (*li)
        *li = (*li)->next;
}

void *
list_reverse_iterator_get_datum(list_reverse_iterator_t *li)
{
    return *li ? (*li)->datum : NULL;
}

bool
list_reverse_iterator_is_valid(list_reverse_iterator_t *li)
{
    return (li != NULL);
}