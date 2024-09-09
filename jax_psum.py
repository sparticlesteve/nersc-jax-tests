import jax
import os
import sys


def run():
    rank = int(os.environ["SLURM_PROCID"])
    print('Rank %d running pid %d'%(rank, os.getpid()))
    address = os.environ['MASTER_ADDR']+':29500'
    print(address)
    jax.distributed.initialize(coordinator_address=address)
    print(jax.device_count())  # total number of accelerator devices in the cluster
    print(jax.local_device_count())  # number of accelerator devices attached to this host
    # The psum is performed over all mapped devices across the pod slice
    xs = jax.numpy.ones(jax.local_device_count())
    print('Performing psum')
    x = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
    print(x)


if __name__ == '__main__':
    run()
