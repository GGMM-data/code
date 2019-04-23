class AgentConfig(object):
    scale = 100

    memory_size = 100 * scale

    batch_size = 32
    state_format = 'NCHW'
    history_length = 4

    action_repeat = True
    random_start = 5
    display = True

    learn_start = 5. * scale


class EnvironmentConfig(object):
    env_name = 'Breakout-v4'

    screen_width = 84
    screen_height = 84


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    pass

class M1(DQNConfig):
    backend = 'tf'


class M2(DQNConfig):
    backend = 'tf'


def get_config(FLAGS):
    if FLAGS.model == 'm1':
        config = M1
    elif FLAGS.model =='m2':
        config = M2
    print(type(FLAGS))
    for k,v in list(FLAGS.flag_values_dict().items()):
        if k == 'gpu':
            if not v:
                config.state_format = 'NHWC'
            else:
                config.state_format = 'NCHW'
        if hasattr(config, k):
            setattr(config, k, v)

    return config
