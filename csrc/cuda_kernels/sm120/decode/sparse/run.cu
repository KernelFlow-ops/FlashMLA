#include "kernel.h"

#include <kerutils/kerutils.cuh>

namespace sm120::decode::sparse {

void run_sparse_decode(const SparseAttnDecodeParams &params) {
    (void)params;
    KU_ASSERT(false, "the sm120 sparse decode kernel is not implemented yet");
}

}
