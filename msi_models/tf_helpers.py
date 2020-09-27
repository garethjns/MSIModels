import tensorflow as tf


def limit_gpu_memory(mb: int = 5000) -> bool:
    try:
        tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=mb)])
        # GPU available, succeeded in setting memory on first device
        return True

    except IndexError:
        # May be no GPU available.
        return False

    except RuntimeError:
        # May be trying to modify existing device , which fails(eg. when running tests.)
        return False
