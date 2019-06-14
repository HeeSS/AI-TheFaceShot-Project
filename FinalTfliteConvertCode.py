from tensorflow.python.tools import freeze_graph
import tensorflow as tf

INPUT_NODE = ["X-input"]
OUTPUT_NODE = ["int32_output"]
input_graph_path = '/graph.pb'
checkpoint_path = '/model.ckpt'
output_node_names = "int32_output"
restore_op_name = "restore_all"
filename_tensor_name = "Const:0"
output_frozen_graph_name = '/frozen_model.pb'

freeze_graph.freeze_graph(input_graph_path, checkpoint_path, output_node_names, restore_op_name, filename_tensor_name, output_frozen_graph_name)
converter = tf.contrib(INPUT_NODE, OUTPUT_NODE, output_frozen_graph_name)
tflite_model = converter.convert()
open("/converted_model.tflite", "wb").write(tflite_model)