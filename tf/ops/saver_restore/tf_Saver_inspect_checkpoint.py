from tensorflow.python.tools import inspect_checkpoint as ckpt


ckpt.print_tensors_in_checkpoint_file("saver/variables/all_variables.ckpt", tensor_name='', all_tensors=True)
