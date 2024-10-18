import jax
import os
import sys


def run():
    rank = int(os.environ["SLURM_PROCID"])
    address = os.environ['MASTER_ADDR']+':29500'
    print(f'Rank {rank} coordinator {address}')
    jax.distributed.initialize(coordinator_address=address)
    print(f'devices {jax.device_count()}', f'local_devices {jax.local_device_count()}')
    # The psum is performed over all mapped devices across the pod slice
    xs = jax.numpy.ones(jax.local_device_count())
    print('Performing psum')
    x = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
    print(x)


if __name__ == '__main__':
    run()
