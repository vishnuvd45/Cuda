#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>

#define False 0
#define True 1
#define BLKDIM 1024
#define N 50000

using clock_value_t = long long;

void srand(unsigned int seed);

double gettime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void swap_cpu(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// bubble sort 
void bubbleSort(int arr[], int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)
        for (j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap_cpu(&arr[j], &arr[j + 1]);
}

void oddEvenSort(int arr[], int n)
{
    bool isSorted = false;

    while (!isSorted)
    {
        isSorted = true;

        // Perform Bubble sort on odd indexed element
        for (int i = 1; i <= n - 2; i = i + 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap_cpu(&arr[i], &arr[i + 1]);
                isSorted = false;
            }
        }

        // Perform Bubble sort on even indexed element
        for (int i = 0; i <= n - 2; i = i + 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap_cpu(&arr[i], &arr[i + 1]);
                isSorted = false;
            }
        }
    }

    return;
}
// end bubble sort

void printArray(int arr[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int arraySortedOrNot(int arr[], int n)
{
    if (n == 1 || n == 0)
        return 1;

    if (arr[n - 1] < arr[n - 2])
        return 0;

    return arraySortedOrNot(arr, n - 1);
}

void copyArray(int arr1[], int arr2[], int n)
{
    for (int i = 0; i < n; i++)
    {
        arr2[i] = arr1[i];
    }
}

bool sameElement(int arr1[], int arr2[], int n)
{
    bool ok = True;
    for (int i = 0; i < n; i++)
    {
        bool single_ok = False;
        for (int j = 0; j < n; j++)
        {
            if (arr1[i] == arr2[j])
            {
                single_ok = True;
            }
        }
        if (single_ok == False)
        {
            ok = False;
        }
    }
    return ok;
}

void init_rand_array(int arr[], int n)
{
    int i;
    for (i = 0; i < n; i++)
        arr[i] = rand() % 100000;
}

__device__ void swap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do
    {
        cycles_elapsed = clock64() - start;
    } while (cycles_elapsed < sleep_cycles);
}

// bubble sort parallel VERSION 1 and 2 ===> Only blocks runned in parallel for 1, the same but with sync() for 2
__global__ void even_only_block(int *arr, int n)
{
    // sleep(10000);

    int index = blockIdx.x;
    index = index * 2;

    if (index <= (n - 2))
    {
        if (arr[index] > arr[index + 1])
        {
            swap(&arr[index], &arr[index + 1]);
        }
    }
}

__global__ void odd_only_block(int *arr, int n)
{
    // sleep(500000);

    int index = blockIdx.x;
    index = index * 2 + 1;

    if (index <= (n - 2))
    {
        if (arr[index] > arr[index + 1])
        {
            swap(&arr[index], &arr[index + 1]);
        }
    }
}
// end of VERSION 1 and 2

// bubble sort parallel VERSION 3
__global__ void even_block_thread(int *arr, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = index * 2;

    if (index <= (n - 2))
    {
        if (arr[index] > arr[index + 1])
        {
            swap(&arr[index], &arr[index + 1]);
        }
    }
}

__global__ void odd_block_thread(int *arr, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = index * 2 + 1;

    if (index <= (n - 2))
    {
        if (arr[index] > arr[index + 1])
        {
            swap(&arr[index], &arr[index + 1]);
        }
    }
}
// end of VERSION 3

int minimum(int x, int y)
{
    return (x < y) ? x : y;
}

// recursive merge sort
void merge_rec(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
    are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
    are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort_rec(int arr[], int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort_rec(arr, l, m);
        mergeSort_rec(arr, m + 1, r);

        merge_rec(arr, l, m, r);
    }
}
// end recursive merge sort

// bottom-up version of merge-sort
void merge_bottom_up(int A[], int temp[], int from, int mid, int to)
{
    int k = from, i = from, j = mid + 1;

    // loop till no elements are left in the left and right runs
    while (i <= mid && j <= to)
    {

        if (A[i] < A[j])
        {
            temp[k++] = A[i++];
        }
        else
        {
            temp[k++] = A[j++];
        }
    }

    // copy remaining elements
    while (i < N && i <= mid)
    {
        temp[k++] = A[i++];
    }

    /* no need to copy the second half (since the remaining items
       are already in their correct position in the temporary array) */

    // copy back to the original array to reflect sorted order
    for (int i = from; i <= to; i++)
    {
        A[i] = temp[i];
    }
}

void mergesort_bottom_up(int A[], int temp[], int low, int high)
{
    // divide the array into blocks of size `m`
    // m = [1, 2, 4, 8, 16…]

    for (int m = 1; m <= high - low; m = 2 * m)
    {

        for (int i = low; i < high; i += 2 * m)
        {
            int from = i;
            int mid = i + m - 1;
            int to = min(i + 2 * m - 1, high);

            merge_bottom_up(A, temp, from, mid, to);
        }
    }
}
// end bottom-up merge-sort

// merge sort parallel version (inspired on bottom-up merge sort)
__device__ void merge_parallel(int *arr, int *temp, int start, int end, int middle, int n)
{
    int k = start, i = start, j = middle + 1;

    // loop till no elements are left in the left and right runs
    while (i <= middle && j <= end)
    {

        if (arr[i] < arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    // copy remaining elements
    while (i < n && i <= middle) ////// cambiato <= di middle in < e basta
    {
        temp[k++] = arr[i++];
    }

    // copy back to the original array to reflect sorted order
    for (int q = start; q <= end; q++)
    {
        arr[q] = temp[q];
    }
}

__global__ void mergeSort_parallel(int *arr, int *temp, int n, int width)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * width * 2;
    if (index < n)
    {
        int start = index;
        int middle = index + width - 1;
        int end = min(index + 2 * width - 1, n - 1);

        // int end = min(start + width - 1, n-1);
        // int middle = (start + end + 1) / 2;

        merge_parallel(arr, temp, start, end, middle, n);
    }
}
// end of merge sort parallel version


int main(void)
{
    srand((unsigned)(42));
    int i, arr1[N], arr2[N], arrCopy[N], temp[N], temp_parallel[N], arr3[N], arr4[N], arr5[N], arr6[N], arr7[N], arr8[N];
    bool sort;
    double tstart, tend;

    init_rand_array(arr1, N);
    copyArray(arr1, arrCopy, N);
    copyArray(arr1, temp, N);
    copyArray(arr1, temp_parallel, N);
    copyArray(arr1, arr2, N);
    copyArray(arr1, arr3, N);
    copyArray(arr1, arr4, N);
    copyArray(arr1, arr5, N);
    copyArray(arr1, arr6, N);
    copyArray(arr1, arr7, N);
    copyArray(arr1, arr8, N);

    printf("\n-----------------------------------------------------------------------------------------------------------\n");

    printf("\n                                    BUBBLE SORT: \n");

    /* Classic bubble sort alg --------------------------------------- */
    printf("--- Classic bubble sort --- \n");
    printf("Array built pre parallel computation: \n");
    printArray(arr1, N);

    tstart = gettime();
    bubbleSort(arr1, N);
    tend = gettime();
    printf("Elapsed time in seconds for bubble sort sequential implementation: %f\n", tend - tstart);

    printf("Sorted array: \n");
    printArray(arr1, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr1, N);
    printf("%d\n", sort);

    /* odd-even sort alg --------------------------------------- */
    printf("\n");
    printf("--- Odd-even sort ---\n");
    printf("Array built pre parallel computation: \n");
    printArray(arr2, N);

    tstart = gettime();
    oddEvenSort(arr2, N);
    tend = gettime();
    printf("Elapsed time in seconds for odd-even sequential implementation: %f\n", tend - tstart);

    printf("Sorted array: \n");
    printArray(arr2, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr2, N);
    printf("%d\n", sort);

    /* bubble sort parallel VERSION 1 --------------------------------------- */
    printf("\n");
    printf("--- Version 1 parallel ---\n");

    int *d_arr; 
    const size_t size = N * sizeof(int);
    cudaMalloc((void **)&d_arr, size);

    printf("Array built pre parallel computation: \n");
    printArray(arr3, N);

    cudaMemcpy(d_arr, arr3, size, cudaMemcpyHostToDevice);

    tstart = gettime();

    for (i = 0; i <= (N / 2); i++)
    {
        even_only_block<<<N / 2, 1>>>(d_arr, N);
        odd_only_block<<<N / 2, 1>>>(d_arr, N);
    }

    tend = gettime();
    printf("Elapsed time in seconds for parallel thread implementation: %f\n", tend - tstart);

    cudaMemcpy(arr3, d_arr, size, cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    printArray(arr3, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr3, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr3, arrCopy, N);
    printf("%d\n", sort);

    cudaFree(d_arr);

    /* bubble sort parallel VERSION 2 --------------------------------------- */
    printf("\n");
    printf("--- Version 2 parallel ---\n");

    int *d_arr2; 
    const size_t size2 = N * sizeof(int);
    cudaMalloc((void **)&d_arr2, size2);

    printf("Array built pre parallel computation: \n");
    printArray(arr4, N);

    cudaMemcpy(d_arr2, arr4, size2, cudaMemcpyHostToDevice);

    tstart = gettime();

    for (i = 0; i <= (N / 2); i++)
    {
        even_only_block<<<N / 2, 1>>>(d_arr2, N);
        cudaDeviceSynchronize();
        odd_only_block<<<N / 2, 1>>>(d_arr2, N);
        cudaDeviceSynchronize();
    }

    tend = gettime();
    printf("Elapsed time in seconds for parallel thread implementation: %f\n", tend - tstart);

    cudaMemcpy(arr4, d_arr2, size2, cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    printArray(arr4, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr4, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr4, arrCopy, N);
    printf("%d\n", sort);

    cudaFree(d_arr2);

    /* bubble sort parallel VERSION 3 --------------------------------------- */
    printf("\n");
    printf("--- Version 3 parallel ---\n");

    int *d_arr3; 
    const size_t size3 = N * sizeof(int);
    cudaMalloc((void **)&d_arr3, size3);

    printf("Array built pre parallel computation: \n");
    printArray(arr5, N);

    cudaMemcpy(d_arr3, arr5, size3, cudaMemcpyHostToDevice);

    tstart = gettime();

    for (i = 0; i <= (N / 2); i++)
    {
        even_block_thread<<<((N / 2) + BLKDIM - 1) / BLKDIM, BLKDIM>>>(d_arr3, N);
        cudaDeviceSynchronize();
        odd_block_thread<<<((N / 2) + BLKDIM - 1) / BLKDIM, BLKDIM>>>(d_arr3, N);
        cudaDeviceSynchronize();
    }

    tend = gettime();
    printf("Elapsed time in seconds for parallel thread implementation: %f\n", tend - tstart);

    cudaMemcpy(arr5, d_arr3, size3, cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    printArray(arr5, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr5, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr5, arrCopy, N);
    printf("%d\n", sort);

    cudaFree(d_arr3);

    printf("\n-----------------------------------------------------------------------------------------------------------\n");

    printf("\n                                      MERGE SORT: ");

    /* recursive merge sort --------------------------------------- */
    printf("\n");
    printf("--- Recursive merge sort ---\n");
    printArray(arr6, N);

    tstart = gettime();
    mergeSort_rec(arr6, 0, N - 1);
    tend = gettime();
    printf("Elapsed time in seconds: %f", tend - tstart);

    printf("\nSorted array is\n");
    printArray(arr6, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr6, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr6, arrCopy, N);
    printf("%d\n", sort);

    /* bottom-up merge sort --------------------------------------- */
    printf("\n");
    printf("--- Bottom-up merge sort ---\n");
    printArray(arr7, N);

    tstart = gettime();
    mergesort_bottom_up(arr7, temp, 0, N - 1);
    tend = gettime();
    printf("Elapsed time in seconds: %f\n", tend - tstart);

    printf("Sorted array is \n");
    printArray(arr7, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr7, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr7, arrCopy, N);
    printf("%d\n", sort);

    /* merge sort parallel version --------------------------------------- */
    printf("\n");
    printf("--- Bottom-up merge sort parallel version ---\n");
    printArray(arr8, N);

    int *d_arr4, *d_temp_parallel; 
    const size_t size4 = N * sizeof(int);
    cudaMalloc((void **)&d_arr4, size4);
    cudaMalloc((void **)&d_temp_parallel, size4);

    cudaMemcpy(d_arr4, arr8, size4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_parallel, temp_parallel, size4, cudaMemcpyHostToDevice);

    tstart = gettime();

    int width = 1;
    while (width <= N)
    {
        mergeSort_parallel<<<((N / width) + BLKDIM - 1) / BLKDIM, BLKDIM>>>(d_arr4, d_temp_parallel, N, width);
        // cudaDeviceSynchronize();
        width = width * 2;
    }

    tend = gettime();
    printf("Elapsed time in seconds for parallel thread implementation: %f\n", tend - tstart);

    cudaMemcpy(arr8, d_arr4, size4, cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    printArray(arr8, N);

    printf("Check if sorted correctly: ");
    sort = arraySortedOrNot(arr8, N);
    printf("%d\n", sort);

    printf("Check if same elements: ");
    sort = sameElement(arr8, arrCopy, N);
    printf("%d\n", sort);

    cudaFree(d_arr4);
    cudaFree(d_temp_parallel);

    return 0;
}