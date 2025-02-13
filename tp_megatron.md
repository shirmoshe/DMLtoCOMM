this file contain functions from megatron github. we analyze these function to understand the parallelism strategy that megatron uses. 

1. 
def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """Sets tp attributes to tensor"""
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)



2. def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, is_expert=False):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not is_expert:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)





   
