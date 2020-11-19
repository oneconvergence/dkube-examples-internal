from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import requests
import json
import os
import logging

class dkubeLoggerHook(tf.estimator.LoggingTensorHook):
        def __init__(self, tensors, every_n_iter=None, every_n_secs=None, at_end=False, formatter=None):
                self.mode = tensors.pop('mode', "NA")
                self.steps_epoch = int(tensors.pop('steps_epoch', 0))
                super(dkubeLoggerHook, self).__init__(tensors, every_n_iter, every_n_secs, at_end, formatter)

        def after_run(self, run_context, run_values):
                super(dkubeLoggerHook, self).after_run(run_context, run_values)
                metrics = run_values.results
                if metrics:
                    step = {
                      'mode': self.mode,
                      'epoch': int(min((metrics['step']/self.steps_epoch + 1), int(os.getenv('EPOCHS', 1)))) if self.steps_epoch > 0 else -1,
                      'key'  :  'step',
                      'value' : int(min(metrics['step'], int(os.getenv('STEPS', 1000)))) if 'step' in metrics else -1,
                      'jobid': os.getenv('DKUBE_JOB_ID'),
                      'run_id': os.getenv('DKUBE_JOB_UUID'),
                      'username': os.getenv('DKUBE_USER_LOGIN_NAME'),
                      'max_steps': os.getenv('STEPS', "-1")
                     }
                    loss = {
                         'mode': self.mode,
                         'key' : 'loss',
                         'value': float(round(metrics['loss'], 4)) if 'loss' and 'step' in metrics else -1,
                         'epoch': int(min((metrics['step']/self.steps_epoch + 1), int(os.getenv('EPOCHS', 1)))) if self.steps_epoch > 0 else -1,
                         'step' : int(min(metrics['step'], int(os.getenv('STEPS', 1000)))) if 'step' in metrics else -1,
                         'jobid': os.getenv('DKUBE_JOB_ID'),
                         'run_id': os.getenv('DKUBE_JOB_UUID'),
                         'username': os.getenv('DKUBE_USER_LOGIN_NAME'),
                         'max_steps': os.getenv('STEPS', "-1")
                      }
                    accuracy = {
                         'mode': self.mode,
                         'key': "accuracy",
                         'value': float(round(metrics['accuracy'], 4)) if 'accuracy' and 'step' in metrics else -1,
                         'epoch': int(min((metrics['step']/self.steps_epoch + 1), int(os.getenv('EPOCHS', 1)))) if self.steps_epoch > 0 else -1,
                         'step' : int(min(metrics['step'], int(os.getenv('STEPS', 1000)))) if 'step' in metrics else -1,
                         'jobid': os.getenv('DKUBE_JOB_ID'),
                         'run_id': os.getenv('DKUBE_JOB_UUID'),
                         'username': os.getenv('DKUBE_USER_LOGIN_NAME'),
                         'max_steps': os.getenv('STEPS', "-1")
                     }
                    try:
                        logging.info("accuracy="+str(float(round(metrics['accuracy'], 4))))
                        logging.info("loss="+str(float(round(metrics['loss'], 4))))
                        url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
                        requests.post(url, json = accuracy )
                        requests.post(url, json = loss )
                        requests.post(url, json = step )
                    except Exception as exc:
                        logging.error(exc)
