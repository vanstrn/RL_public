# Module contains any methods, class, parameters, etc that is related to logging the trainig

import tensorflow as tf
import numpy as np

def record(item, writer, step):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary, step)
    writer.flush()

def RecordError(writer,msg):
    text_tensor = tf.make_tensor_proto(msg, dtype=tf.string)
    summary = tf.Summary()
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary.value.add(tag="Error Logs", metadata=meta, tensor=text_tensor)
    writer.add_summary(summary)
    writer.flush()

def RecordDescription(writer,msg):
    text_tensor = tf.make_tensor_proto(msg, dtype=tf.string)
    summary = tf.Summary()
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary.value.add(tag="Description", metadata=meta, tensor=text_tensor)
    writer.add_summary(summary)
    writer.flush()

def SaveHyperparams(writer,hParams):
    """
    hParams : Dictionary Containing Labeled Hyperparameters.
    """
    text = ""
    for k,v in hParams.items():
        text = text + k + " | " +str(v) + "\n\n"

    text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
    summary = tf.Summary()
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary.value.add(tag="Hyperparameters", metadata=meta, tensor=text_tensor)
    writer.add_summary(summary)
    writer.flush()

if __name__ == "__main__":
    logdir =  '/home/capturetheflag/RL/logs/Test1'
    writer = tf.compat.v1.summary.FileWriter(logdir)
    RecordDescription(writer,"This is a description of the experiment that was run.")
    hParams = {
        "Epsilon": 0.99,
        "Gamma" : 0.5,
        }
    SaveHyperparams(writer,hParams)
